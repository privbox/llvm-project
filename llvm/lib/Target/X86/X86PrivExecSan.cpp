#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<bool>
    EnablePrivSan("enable-priv-san", cl::NotHidden,
                     cl::desc("X86: Sanitize for privileged execution."),
                     cl::init(false));
cl::opt<unsigned int>
    PrivSanBtrBit("priv-san-btr-bit", cl::NotHidden,
                     cl::desc("X86: Sanitize for privileged execution - high bit to clear."),
                     cl::init(50));
cl::opt<unsigned int>
    PrivSanAlignment("priv-san-align-bytes", cl::NotHidden,
                     cl::desc("X86: alignment for branch sanitization."),
                     cl::init(32));