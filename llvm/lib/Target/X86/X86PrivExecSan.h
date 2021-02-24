#ifndef LLVM_LIB_TARGET_X86_X86PRIVSAN_H
#define LLVM_LIB_TARGET_X86_X86PRIVSAN_H

#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define MARKER_MAGIC 0x12345678

extern cl::opt<unsigned> PrivSanBtrBit;
extern cl::opt<unsigned> PrivSanAlignment;
extern cl::opt<bool> EnablePrivSan;
extern cl::opt<bool> EnablePrivSanLS;
extern cl::opt<bool> EnablePrivSanBr;
extern cl::opt<bool> EnablePrivSanSpar;

class RegisterGetter {
protected:
  virtual ~RegisterGetter() {}

public:
  virtual Register get() = 0;
};

class StaticRegisterGetter : public RegisterGetter {
  Register reg;

public:
  StaticRegisterGetter(Register reg) : reg(reg) {}
  virtual ~StaticRegisterGetter() {}
  virtual Register get() override { return reg; }
};

class AllocatingRegisterGetter : public RegisterGetter {
  MachineRegisterInfo *MRI;
  const TargetRegisterClass *RegClass;

public:
  AllocatingRegisterGetter(MachineRegisterInfo *MRI,
                           const TargetRegisterClass *RegClass)
      : MRI(MRI), RegClass(RegClass) {}
  virtual ~AllocatingRegisterGetter() {}
  virtual Register get() override {
    return MRI->createVirtualRegister(RegClass);
  }
};

class X86PrivSanBase : public MachineFunctionPass {
protected:
  const X86Subtarget *Subtarget = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const X86InstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  void setupOnFunc(MachineFunction &MF) {
    Subtarget = &MF.getSubtarget<X86Subtarget>();
    MRI = &MF.getRegInfo();
    TII = Subtarget->getInstrInfo();
    TRI = Subtarget->getRegisterInfo();
  }

  virtual void instructionEmitted(MachineInstr *MI) {
    LLVM_DEBUG(dbgs() << *MI);
  }

  bool isEFLAGSLive(MachineInstr &MI) {
    MachineBasicBlock &MBB = *MI.getParent();
    MachineBasicBlock::iterator I = MI.getIterator();
    // Check if EFLAGS are alive by seeing if there is a def of them or they
    // live-in, and then seeing if that def is in turn used.
    for (MachineInstr &MI : llvm::reverse(llvm::make_range(MBB.begin(), I))) {
      if (MachineOperand *DefOp = MI.findRegisterDefOperand(X86::EFLAGS)) {
        // If the def is dead, then EFLAGS is not live.
        if (DefOp->isDead())
          return false;

        // Otherwise we've def'ed it, and it is live.
        return true;
      }
      // While at this instruction, also check if we use and kill EFLAGS
      // which means it isn't live.
      if (MI.killsRegister(X86::EFLAGS, TRI))
        return false;
    }

    // If we didn't find anything conclusive (neither definitely alive or
    // definitely dead) return whether it lives into the block.
    return MBB.isLiveIn(X86::EFLAGS);
  }

  /// Save EFLAGS into the returned GPR. This can in turn be restored with
  /// `restoreEFLAGS`.
  ///
  /// Note that LLVM can only lower very simple patterns of saved and restored
  /// EFLAGS registers. The restore should always be within the same basic block
  /// as the save so that no PHI nodes are inserted.
  std::pair<MachineInstr *, unsigned>
  saveEFLAGS(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
             DebugLoc Loc) {
    // FIXME: Hard coding this to a 32-bit register class seems weird, but
    // matches what instruction selection does.
    Register Reg = MRI->createVirtualRegister(&X86::GR32RegClass);
    // We directly copy the FLAGS register and rely on later lowering to clean
    // this up into the appropriate setCC instructions.
    auto CopyI = BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), Reg)
                     .addReg(X86::EFLAGS);
    return std::make_pair(CopyI, Reg);
  }

  /// Restore EFLAGS from the provided GPR. This should be produced by
  /// `saveEFLAGS`.
  ///
  /// This must be done within the same basic block as the save in order to
  /// reliably lower.
  MachineInstr *restoreEFLAGS(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator InsertPt,
                              DebugLoc Loc, Register Reg) {
    return BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), X86::EFLAGS)
        .addReg(Reg);
  }

  MachineInstr *emitMarker(MachineInstr &MI) {
    MachineBasicBlock &MBB = *MI.getParent();
    return BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                   TII->get(X86::NOOPQ))
        .addReg(X86::NoRegister)
        .addImm(1)
        .addReg(X86::NoRegister)
        .addImm(MARKER_MAGIC)
        .addReg(X86::NoRegister);
  }

  bool isMarker(const MachineInstr &MI) {
    return (MI.getOpcode() == X86::NOOPQ &&
            MI.getOperand(3).getImm() == MARKER_MAGIC);
  }

  Register LeaReg(MachineInstr &MI, RegisterGetter &RG) {
    MachineBasicBlock &MBB = *MI.getParent();
    const MCInstrDesc &Desc = MI.getDesc();
    int MemRefBeginIdx =
        X86II::getMemoryOperandNo(Desc.TSFlags) + X86II::getOperandBias(Desc);

    Register resReg = RG.get();
    auto LeaI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::LEA64r), resReg)
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrDisp))
                    .add(MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg));
    instructionEmitted(LeaI);
    return resReg;
  }

  Register BtrRReg(Register inReg, MachineInstr &MI, RegisterGetter &RG) {
    MachineBasicBlock &MBB = *MI.getParent();
    Register resReg = RG.get();
    auto BtrI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::BTR64ri8), resReg)
                    .addReg(inReg)
                    .addImm(PrivSanBtrBit);
    instructionEmitted(BtrI);
    return resReg;
  }

  Register SanLoad(MachineInstr &MI, RegisterGetter &RG) {
    Register resReg = RG.get();

    if (EnablePrivSanLS) {
      Register ptrReg = LeaReg(MI, RG);
      Register maskedPtrReg = BtrRReg(ptrReg, MI, RG);

      MachineBasicBlock &MBB = *MI.getParent();
      auto MovI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII->get(X86::MOV64rm), resReg)
                      .addReg(maskedPtrReg)
                      .addImm(1)
                      .addReg(X86::NoRegister)
                      .addImm(0)
                      .addReg(X86::NoRegister);
      instructionEmitted(MovI);
    } else {
      MachineBasicBlock &MBB = *MI.getParent();
      const MCInstrDesc &Desc = MI.getDesc();
      int MemRefBeginIdx =
          X86II::getMemoryOperandNo(Desc.TSFlags) + X86II::getOperandBias(Desc);

      auto MovI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII->get(X86::MOV64rm), resReg)
                      .add(MI.getOperand(MemRefBeginIdx + X86::AddrBaseReg))
                      .add(MI.getOperand(MemRefBeginIdx + X86::AddrScaleAmt))
                      .add(MI.getOperand(MemRefBeginIdx + X86::AddrIndexReg))
                      .add(MI.getOperand(MemRefBeginIdx + X86::AddrDisp))
                      .add(MI.getOperand(MemRefBeginIdx + X86::AddrSegmentReg));
      instructionEmitted(MovI);
    }
    return resReg;
  }

  Register AndF0Reg(Register inReg, MachineInstr &MI, RegisterGetter &RG) {
    MachineBasicBlock &MBB = *MI.getParent();
    Register resReg = RG.get();

    auto AndI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII->get(X86::AND64ri8), resReg)
                    .addReg(inReg)
                    .addImm(~((PrivSanAlignment)-1));
    instructionEmitted(AndI);
    return resReg;
  }

  Register BrSanReg(Register inReg, MachineInstr &MI, RegisterGetter &RG,
                    bool alignUp = false) {
    MachineBasicBlock &MBB = *MI.getParent();
    Register tmpReg;

    if (alignUp) {
      auto AddI = BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII->get(X86::ADD64ri8), tmpReg)
                      .addReg(inReg)
                      .addImm(PrivSanAlignment - 1);
      instructionEmitted(AddI);
    } else {
      tmpReg = inReg;
    }

    Register alignedReg = AndF0Reg(inReg, MI, RG);

    Register resReg;
    if (EnablePrivSanLS) {
      resReg = BtrRReg(alignedReg, MI, RG);
    } else {
      resReg = alignedReg;
    }
    return resReg;
  }

  void alignMI(MachineInstr &MI, bool UpdateLiveIns = false) {
    MachineBasicBlock *MBB = MI.getParent();
    MachineBasicBlock *TargetMBB = MBB;
    MachineBasicBlock::iterator I = MI.getIterator();
    if (I != MBB->begin()) {
      --I;
      MachineInstr &PrevMI = *I;
      TargetMBB = MBB->splitAt(PrevMI, UpdateLiveIns, NULL);
    }
    TargetMBB->setAlignment(Align(PrivSanAlignment));
  }

public:
  X86PrivSanBase(char &ID) : MachineFunctionPass(ID) {}
};

#endif // LLVM_LIB_TARGET_X86_X86PRIVSAN_H