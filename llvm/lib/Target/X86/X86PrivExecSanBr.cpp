#include "X86.h"

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86FrameLowering.h"

#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "priv-san"

#include "X86PrivExecSan.h"

/*
Transformations:

2. Branches
* Branch targets must be MSB=0 and aligned.
* Return must be altered after epilogue is emitted
* Return can use real call-clobbered register, e.g. RSI.
* Call/Return can ignore flags.
* Jumps must preserve flags if they're live.

Instruction:
        %rax = ret
Transformation:
        Use RSI to pop, mask and branch.
        3 extra instructions, 2 if we have a global mask

Alterative 1:
        pop %rsi
        btr $63, %rsi
        %rsi = and $((1<<align)-1), %rsi
******* jmp *%rsi

Alternative 2:
        pop %rsi
        and MASK_SYMBOL, %rsi
******* jmp *%rsi

Instruction:
        %res = call $disp(%base, $scale, %index)
Transformation:
    4-5 extra instructions

.if both [%base, %index] are set
        %Tmp1 = lea $disp(%base, $scale, %index)
        %Tmp2 = btr $63, %Tmp1
.else
        %Tmp2 = btr $63, %[base|index]
.end
        %Addr = and $((1<<align)-1), %Tmp2
******* %res = call *%Addr

Instruction:
        jmpCC $disp(%base, $scale, %index)

Transformation:
    2-5 extra instructions

.if EFLAGS live at instruction
        %FlagsReg = COPY %EFLAGS
.end

.if both [%base, %index] are set
        %Tmp1 = leaq $disp(%base, $scale, %index)
        %Addr = btr $63, %Tmp1
.else
        %Addr = btr $63, %[base|index]
.end
        %Addr = and $((1<<align)-1), %Tmp2

.if EFLAGS were saved
        %EFLAGS = COPY %FlagsReg  (if were EFLAGS saved)
.end

******* jmpCC *%Addr

*/

using namespace llvm;

STATISTIC(NumPrivSanBrAdded,
          "Number of branching instrumentation instructions added");
STATISTIC(NumPrivSanBrVulnRet, "Number of vulnerable returns");
STATISTIC(NumPrivSanBrVulnCallsites, "Number of detected callsites");

cl::opt<bool> EnablePrivSanBr(
    "enable-priv-san-branching", cl::NotHidden,
    cl::desc(
        "X86: Sanitize for privileged execution (branching instrumentatation)"),
    cl::init(false));

cl::opt<bool> EnablePrivSanCallJmp(
    "enable-priv-san-calljmp", cl::NotHidden,
    cl::desc(
        "X86: Sanitize for privileged execution (branching instrumentatation)"),
    cl::init(false));

cl::opt<bool> EnablePrivSanRet(
    "enable-priv-san-ret", cl::NotHidden,
    cl::desc("X86: Sanitize for privileged execution (return)"),
    cl::init(false));

namespace {

enum HardenType {
  HardenNone,
  HardenCallM,
  HardenCallR,
  HardenBranchM,
  HardenBranchR,
};

struct HardenInfo {
  MachineInstr *MI;
  HardenType hardenType;

  HardenInfo(MachineInstr *MI, HardenType type) : MI(MI), hardenType(type) {}
};

class X86PrivExecBrSanitizer : public X86PrivSanBase {
  // AllocatingRegisterGetter RG;

  AllocatingRegisterGetter RG() {
    return AllocatingRegisterGetter(MRI, &X86::GR64RegClass);
  }
  virtual void instructionEmitted(MachineInstr *MI) override {
    X86PrivSanBase::instructionEmitted(MI);
    NumPrivSanBrAdded++;
  }

  HardenType vulnerableCall(MachineInstr &MI) {
    if (!MI.isCall() || MI.isReturn())
      return HardenNone;

    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);

    if (MemRefBeginIdx >= 0)
      return HardenCallM;
    else {
      if (MI.getOpcode() == X86::CALL64r) {
        return HardenCallR;
      }
      if (MI.getOpcode() != X86::CALL64pcrel32) {
        /*LLVM_DEBUG(*/ outs() << "Unhandled call " << MI /*)*/;
      }
      return HardenNone;
    }
  }

  HardenType vulnerableBranch(MachineInstr &MI) {
    if (!MI.isBranch() || MI.isCall())
      return HardenNone;

    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);

    if (MemRefBeginIdx >= 0) {
      return HardenBranchM;
    }
    return HardenNone;
  }

  void cleanupReplacedInstr(MachineInstr &MI) {
    MI.removeFromParent();
    NumPrivSanBrAdded--;
  }

  void hardenCall(MachineInstr &MI, Register inReg) {
    AllocatingRegisterGetter RG_ = RG();
    MachineBasicBlock &MBB = *MI.getParent();

    Register sanReg = BrSanReg(inReg, MI, RG_);

    auto NewCallI =
        BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII->get(X86::CALL64r))
            .addReg(sanReg);
    NewCallI->copyImplicitOps(*MBB.getParent(), MI);
    instructionEmitted(NewCallI);

    cleanupReplacedInstr(MI);
  }

  void hardenCallM(MachineInstr &MI) {
    AllocatingRegisterGetter RG_ = RG();
    Register addrReg = SanLoad(MI, RG_);
    hardenCall(MI, addrReg);
  }

  void hardenCallR(MachineInstr &MI) {
    hardenCall(MI, MI.getOperand(0).getReg());
  }

  Register saveEFLAGS(MachineInstr &MI) {
    if (!isEFLAGSLive(MI))
      return X86::NoRegister;

    MachineBasicBlock &MBB = *MI.getParent();
    auto pair =
        X86PrivSanBase::saveEFLAGS(MBB, MI.getIterator(), MI.getDebugLoc());
    instructionEmitted(pair.first);
    return pair.second;
  }

  void restoreEFLAGS(MachineInstr &MI, Register eflagsReg) {
    if (eflagsReg == X86::NoRegister)
      return;

    MachineBasicBlock &MBB = *MI.getParent();
    auto CopyI = X86PrivSanBase::restoreEFLAGS(MBB, MI.getIterator(),
                                               MI.getDebugLoc(), eflagsReg);
    instructionEmitted(CopyI);
  }

  void hardenBranch(MachineInstr &MI, Register inReg, Register eflagsReg) {
    AllocatingRegisterGetter RG_ = RG();
    Register sanReg = BrSanReg(inReg, MI, RG_);
    restoreEFLAGS(MI, eflagsReg);

    MachineBasicBlock &MBB = *MI.getParent();
    auto JmpI =
        BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII->get(X86::JMP64r))
            .addReg(sanReg);
    JmpI->copyImplicitOps(*MBB.getParent(), MI);
    instructionEmitted(JmpI);

    cleanupReplacedInstr(MI);
  }

  void hardenBranchM(MachineInstr &MI) {
    AllocatingRegisterGetter RG_ = RG();
    Register eflagsReg = saveEFLAGS(MI);
    Register addrReg = SanLoad(MI, RG_);
    hardenBranch(MI, addrReg, eflagsReg);
  }
  void hardenBranchR(MachineInstr &MI) {
    Register eflagsReg = saveEFLAGS(MI);
    hardenBranch(MI, MI.getOperand(0).getReg(), eflagsReg);
  }

  void harden(HardenType type, MachineInstr &MI) {
    switch (type) {
    case HardenCallM:
      hardenCallM(MI);
      break;
    case HardenBranchM:
      hardenBranchM(MI);
      break;
    case HardenCallR:
      hardenCallR(MI);
      break;
    case HardenBranchR:
      hardenBranchR(MI);
      break;
    case HardenNone:
      break;
    }
  }

  void findHardeningTasks(
      MachineFunction &MF,
      SmallVector<std::pair<HardenType, MachineInstr *>> &tasks) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        HardenType type = vulnerableCall(MI);
        if (type == HardenNone)
          type = vulnerableBranch(MI);

        if (type == HardenNone)
          continue;

        tasks.push_back(std::make_pair(type, &MI));
      }
    }
  }

public:
  static char ID;

  X86PrivExecBrSanitizer() : X86PrivSanBase(ID) {
    initializeX86PrivExecBrSanitizerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!(EnablePrivSanCallJmp || EnablePrivSanBr || EnablePrivSan))
      return false;

    setupOnFunc(MF);
    MF.setAlignment(Align(PrivSanAlignment));

    // FIXME
    outs() << "Func: " << MF.getName() << "\n";

    SmallVector<std::pair<HardenType, MachineInstr *>> tasks;
    findHardeningTasks(MF, tasks);

    bool needAlignMBB = false;
    for (auto &P : tasks) {
      emitMarker(*P.second);
      harden(P.first, *P.second);
      if (P.first == HardenType::HardenBranchM ||
          P.first == HardenType::HardenBranchR)
        needAlignMBB = true;
    }

    if (needAlignMBB)
      for (auto &MBB : MF)
        MBB.setAlignment(Align(PrivSanAlignment));

    return tasks.size() > 0 || needAlignMBB;
  }

  StringRef getPassName() const override { return "PrivExecSanBr"; }
};

char X86PrivExecBrSanitizer::ID = 0;

/* ######################################################################## */
/* Post regs */

class X86PrivExecBrPostRegsSanitizer : public X86PrivSanBase {
  const Register TARGET_REG;
  virtual void instructionEmitted(MachineInstr *MI) override {
    X86PrivSanBase::instructionEmitted(MI);
    NumPrivSanBrAdded++;
  }

  void sanitizeReg(MachineInstr &MI, Register reg) {
    StaticRegisterGetter RG(reg);
    BrSanReg(reg, MI, RG, true);
  }

  void tailJmp(MachineInstr &MI, Register dest) {
    sanitizeReg(MI, dest);

    MachineBasicBlock &MBB = *MI.getParent();
    auto TailJmpI =
        BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII->get(X86::JMP64r))
            .addReg(dest);
    instructionEmitted(TailJmpI);

    MI.removeFromParent();
    NumPrivSanBrAdded--;
  }

  void hardenReturnJmpR(MachineInstr &MI) {
    outs() << "TailcallR!\n";
    alignMI(MI); // FIXME maybe do a single align pre pop
    tailJmp(MI, MI.getOperand(0).getReg());
  }

  void hardenReturnJmpM(MachineInstr &MI) {
    Register targetReg = TARGET_REG;
    outs() << "TailcallM!\n";

    auto LeaR = BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::LEA64r), targetReg)
                    .add(MI.getOperand(X86::AddrBaseReg))
                    .add(MI.getOperand(X86::AddrScaleAmt))
                    .add(MI.getOperand(X86::AddrIndexReg))
                    .add(MI.getOperand(X86::AddrDisp))
                    .add(MI.getOperand(X86::AddrSegmentReg));
    instructionEmitted(LeaR);

    alignMI(MI);
    MachineBasicBlock &MBB = *MI.getParent();

    if (EnablePrivSanLS) {
      auto BtrI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII->get(X86::BTR64ri8), targetReg)
                      .addReg(targetReg)
                      .addImm(PrivSanBtrBit);
      instructionEmitted(BtrI);
    }

    // FIXME unoptimal
    auto MovI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::MOV64rm), targetReg)
                    .addReg(targetReg)
                    .addImm(1)
                    .addReg(X86::NoRegister)
                    .addImm(0)
                    .addReg(X86::NoRegister);
    instructionEmitted(MovI);

    tailJmp(MI, targetReg);
  }

  void hardenReturnRet(MachineInstr &MI) {
    Register targetReg = TARGET_REG;

    auto PopI = BuildMI(*MI.getParent(), MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::POP64r), targetReg);
    instructionEmitted(PopI);

    alignMI(MI);
    tailJmp(MI, targetReg);
  }

  void hardenReturn(MachineInstr &MI) {
    LLVM_DEBUG(dbgs() << "Harden return: " << MI);

    switch (MI.getOpcode()) {
    case X86::TAILJMPr64:
      hardenReturnJmpR(MI);
      break;
    case X86::TAILJMPm64:
      hardenReturnJmpM(MI);
      break;
    default:
      hardenReturnRet(MI);
      break;
    }
    LLVM_DEBUG(dbgs() << "\n");
  }

  void collectReturns(MachineFunction &MF,
                      SmallVector<MachineInstr *> &Result) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (!MI.isReturn()) {
          continue;
        }
        switch (MI.getOpcode()) {
        case X86::TAILJMPd:
        case X86::TAILJMPd64:
        case X86::TAILJMPd_CC:
        case X86::TAILJMPd64_CC:
          continue;
        case X86::TAILJMPr:
        case X86::TAILJMPr64_REX:
        case X86::TAILJMPm:
        case X86::TAILJMPm64_REX:
          assert(0 && "Unhandled cases");
        }

        Result.push_back(&MI);
        NumPrivSanBrVulnRet++;
        LLVM_DEBUG(dbgs() << "Vulnerable return: " << MI);
      }
    }
  }

  void handleReturns(SmallVector<MachineInstr *> &Returns) {
    for (auto MI : Returns) {
      // alignMI(*MI);
      hardenReturn(*MI);
    }
  }

  void collectCalls(MachineFunction &MF, SmallVector<MachineInstr *> &Result) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (MI.isCall() && !MI.isReturn()) {
          Result.push_back(&MI);
          NumPrivSanBrVulnCallsites++;
          LLVM_DEBUG(dbgs() << "Vulnerable call: " << MI);
        } else if (!MI.isPseudo()) {
          LLVM_DEBUG(dbgs() << "Non-call: " << MI);
        }
      }
    }
  }

  void splitMBB(MachineInstr &MI) {
    MachineBasicBlock &MBB = *MI.getParent();
    LLVM_DEBUG(dbgs() << "Original MBB: " << MBB);
    MBB.splitAt(MI, false, nullptr);
    for (auto *SuccMBB : MBB.successors()) {
      LLVM_DEBUG(dbgs() << "Align MBB: " << *SuccMBB);
      SuccMBB->setAlignment(Align(PrivSanAlignment));
    }
  }

  void handleCalls(SmallVector<MachineInstr *> &Calls) {
    for (auto MI : Calls) {
      LLVM_DEBUG(dbgs() << "Harden call with MBB split: " << *MI);
      splitMBB(*MI);
    }
  }

  void collectMarkers(MachineFunction &MF,
                      SmallVector<MachineInstr *> &Result) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (isMarker(MI))
          Result.push_back(&MI);
      }
    }
  }

  void handleMarkers(SmallVector<MachineInstr *> &Markers) {
    for (auto MI : Markers) {
      alignMI(*MI);
      MI->removeFromParent();
    }
  }

public:
  static char ID;

  X86PrivExecBrPostRegsSanitizer() : X86PrivSanBase(ID), TARGET_REG(X86::RCX) {
    initializeX86PrivExecBrPostRegsSanitizerPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    setupOnFunc(MF);
    LLVM_DEBUG(dbgs() << getPassName() << "::" << MF.getName() << "\n");

    SmallVector<MachineInstr *> Returns;
    if (EnablePrivSanRet || EnablePrivSanBr || EnablePrivSan) {
      collectReturns(MF, Returns);
      handleReturns(Returns);
    }

    SmallVector<MachineInstr *> Calls;
    if (EnablePrivSanCallJmp || EnablePrivSanBr || EnablePrivSan) {
      collectCalls(MF, Calls);
      handleCalls(Calls);
    }

    SmallVector<MachineInstr *> Markers;
    collectMarkers(MF, Markers);
    handleMarkers(Markers);

    return (Returns.size() + Calls.size() + Markers.size()) > 0;
  }

  StringRef getPassName() const override { return "PrivExecSanBrPostRegs"; }
};

char X86PrivExecBrPostRegsSanitizer::ID = 0;
} // end of anonymous namespace

INITIALIZE_PASS(X86PrivExecBrSanitizer, "x86-privexecsan-br",
                "Privileged execution sanitizer - branching",
                false, // is CFG only?
                false  // is analysis?
)
INITIALIZE_PASS(X86PrivExecBrPostRegsSanitizer, "x86-privexecsan-br-pr",
                "Privileged execution sanitizer - branching (post regs)",
                false, // is CFG only?
                false  // is analysis?
)

namespace llvm {
FunctionPass *createX86PrivExecBrSanitizerPass() {
  return new X86PrivExecBrSanitizer();
}
FunctionPass *createX86PrivExecBrPostRegsSanitizerPass() {
  return new X86PrivExecBrPostRegsSanitizer();
}
} // namespace llvm