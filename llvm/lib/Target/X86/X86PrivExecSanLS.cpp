#include "X86.h"

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86FrameLowering.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"

#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "priv-san"
#include "X86PrivExecSan.h"

/*
Transformations:

1. Load/Store:
* Valid loads/stores are any addresses with MSB=0
* Must preserve flags if they are live at instruction

Instruction:
    %target = OP $disp(%base, $scale, %index)

Alternative 1:
  Use SH{R,L}X to mask out MSB
  3-4 extra instructions.

        %BitReg = mov $1
.if both [%base, %index] are set
        %Tmp1 = leaq $disp(%base, $scale, %index)
        %Tmp2 = shlx %Tmp1, %BitReg
.else
        %Tmp2 = shlx %[base|index], %BitReg
.end
        %Addr = shlx %Tmp2, %BitReg
******* %target = OP *%Addr

Alternative 2:
  Use BTR to clear MSB, trace EFLAGS liveness to check if they need to be saved.
  1-4 extra instructions.

.if EFLAGS live at instruction
        %FlagsReg = COPY %EFLAGS
.end

.if both [%base, %index] are set
        %Tmp1 = leaq $disp(%base, $scale, %index)
        %Addr = btr $63, %Tmp1
.else
        %Addr = btr $59, %[base|index]
.end

.if EFLAGS were saved
        %EFLAGS = COPY %FlagsReg  (if were EFLAGS saved)
.end

******** %target = OP *%Addr

*/


using namespace llvm;

STATISTIC(NumPrivSanLSAdded,
          "Number of load/store instrumentation instructions added");
STATISTIC(NumPrivSanLSVuln, "Number of vulnerable loads/stores");

cl::opt<bool> EnablePrivSanLS("enable-priv-san-load-store", cl::NotHidden,
                              cl::desc("X86: Sanitize for privileged execution "
                                       "(load-store instrumentatation)"),
                              cl::init(false));

namespace {

class X86PrivExecLSSanitizer : public X86PrivSanBase {
protected:
  bool vulnerableLoadStore(MachineInstr &MI) {
    if (!MI.mayLoadOrStore() || MI.isCall() || MI.isBranch() ||
        MI.getOpcode() == X86::MFENCE || MI.getOpcode() == X86::LFENCE)
      return false;

    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
    if (MemRefBeginIdx < 0) {
      return false;
    }
    MemRefBeginIdx += X86II::getOperandBias(Desc);
    MachineOperand &BaseMO = MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
    MachineOperand &IndexMO = MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);

    unsigned BaseReg = 0, IndexReg = 0;
    if (!BaseMO.isFI() && BaseMO.getReg() != X86::RIP &&
        BaseMO.getReg() != X86::RSP && BaseMO.getReg() != X86::NoRegister)
      BaseReg = BaseMO.getReg();
    if (IndexMO.getReg() != X86::NoRegister)
      IndexReg = IndexMO.getReg();

    if (!BaseReg && !IndexReg) {
      return false;
    }
    return true;
  }

  void hardenLoadStoreOneReg(MachineInstr &MI, MachineOperand &MO) {
    MachineBasicBlock &MBB = *MI.getParent();

    Register maskedReg = MRI->createVirtualRegister(&X86::GR64RegClass);
    auto BtrI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::BTR64ri8), maskedReg)
                    .addReg(MO.getReg())
                    .addImm(60);
    NumPrivSanLSAdded++;
    LLVM_DEBUG(dbgs() << *BtrI);

    MO.setReg(maskedReg);
  }

  void hardenLoadStoreLea(MachineInstr &MI) {
    MachineBasicBlock &MBB = *MI.getParent();
    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx =
        X86II::getMemoryOperandNo(Desc.TSFlags) + X86II::getOperandBias(Desc);

    Register addrReg = MRI->createVirtualRegister(&X86::GR64RegClass);
    auto LeaI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::LEA64r), addrReg)
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrDisp))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg));
    NumPrivSanLSAdded++;
    LLVM_DEBUG(dbgs() << *LeaI);

    Register maskedReg = MRI->createVirtualRegister(&X86::GR64RegClass);
    auto BtrI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::BTR64ri8), maskedReg)
                    .addReg(addrReg)
                    .addImm(PrivSanBtrBit);
    NumPrivSanLSAdded++;
    LLVM_DEBUG(dbgs() << *BtrI);

    if (!MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).isReg()) {
      MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg) =
          MachineOperand(MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg));
    }
    MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg).setReg(maskedReg);
    MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt).setImm(1);
    MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg).setReg(X86::NoRegister);
    MI.getOperand(MemRefBeginIdx + X86::AddrDisp) =
        MachineOperand(MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt));
    MI.getOperand(MemRefBeginIdx + X86::AddrDisp).setImm(0);
    MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg).setReg(X86::NoRegister);
  }

  void harden(MachineInstr &MI) {
    MachineBasicBlock &MBB = *MI.getParent();
    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
    if (MemRefBeginIdx < 0)
      return;

    MemRefBeginIdx += X86II::getOperandBias(Desc);
    MachineOperand &BaseMO = MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg);
    MachineOperand &IndexMO = MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg);

    unsigned BaseReg = 0, IndexReg = 0;
    if (!BaseMO.isFI() && BaseMO.getReg() != X86::RIP &&
        BaseMO.getReg() != X86::RSP && BaseMO.getReg() != X86::NoRegister)
      BaseReg = BaseMO.getReg();
    if (IndexMO.getReg() != X86::NoRegister)
      IndexReg = IndexMO.getReg();

    if (!BaseReg && !IndexReg)
      return;

    LLVM_DEBUG(dbgs() << "Hardening: " << MI);

    Register eflagsReg = X86::NoRegister;
    if (isEFLAGSLive(MI)) {
      std::pair<MachineInstr *, unsigned> res =
          saveEFLAGS(MBB, MI.getIterator(), MI.getDebugLoc());
      eflagsReg = res.second;
      NumPrivSanLSAdded++;
      LLVM_DEBUG(dbgs() << *res.first);
    }

    if ((BaseReg && !IndexReg) &&
        (MI.getOperand(MemRefBeginIdx + X86::AddrDisp).isImm() &&
         MI.getOperand(MemRefBeginIdx + X86::AddrDisp).getImm() == 0) &&
        MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg).getReg() ==
            X86::NoRegister) {
      hardenLoadStoreOneReg(MI, BaseMO);
    } else if ((!BaseReg && IndexReg) &&
               (MI.getOperand(MemRefBeginIdx + X86::AddrDisp).isImm() &&
                MI.getOperand(MemRefBeginIdx + X86::AddrDisp).getImm() == 0) &&
               MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg).getReg() ==
                   X86::NoRegister) {
      hardenLoadStoreOneReg(MI, IndexMO);
    } else {
      hardenLoadStoreLea(MI);
    }

    if (eflagsReg != X86::NoRegister) {
      auto CopyI =
          restoreEFLAGS(MBB, MI.getIterator(), MI.getDebugLoc(), eflagsReg);
      NumPrivSanLSAdded++;
      LLVM_DEBUG(dbgs() << *CopyI);
    }
    LLVM_DEBUG(dbgs() << MI << "\n");
  }

  void findHardeningTasks(MachineFunction &MF,
                          SmallVector<MachineInstr *> &tasks) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (vulnerableLoadStore(MI)) {
          LLVM_DEBUG(dbgs() << "Vulnerable load/store " << MI);
          NumPrivSanLSVuln++;
          tasks.push_back(&MI);
        }
      }
    }
  }

public:
  static char ID;

  X86PrivExecLSSanitizer() : X86PrivSanBase(ID) {
    initializeX86PrivExecLSSanitizerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!(EnablePrivSanLS || EnablePrivSan))
      return false;

    setupOnFunc(MF);
    SmallVector<MachineInstr *> tasks;

    findHardeningTasks(MF, tasks);

    for (auto *MI : tasks) {
      emitMarker(*MI);
      harden(*MI);
    }

    return tasks.size() > 0;
  }

  StringRef getPassName() const override { return "PrivExecSanLS"; }
};

char X86PrivExecLSSanitizer::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(X86PrivExecLSSanitizer, "x86-privexecsan-ls",
                "Privileged execution sanitizer (load-store)",
                false, // is CFG only?
                false  // is analysis?
)

namespace llvm {
FunctionPass *createX86PrivExecLSSanitizerPass() {
  return new X86PrivExecLSSanitizer();
}
} // namespace llvm