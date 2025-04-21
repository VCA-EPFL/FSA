package msaga.sa

import chisel3._
import chisel3.util._

class SIntMacUnit(elemWidth: Int, accWidth: Int) extends
  MacUnit[SInt, SInt](SInt(elemWidth.W), SInt(accWidth.W))
{
  io.out := Mux(io.in_cmd === MacCMD.MAC,
    (io.in_a * io.in_b) +& io.in_c,
    io.in_a +& 1.S // FIXME: for testing only
  )
}

class SIntCmpUnit(width: Int) extends CmpUnit(SInt(width.W)) {
  io.out := Mux(io.in_cmd === CmpCMD.MAX,
    Mux(io.in_a > io.in_b, io.in_a, io.in_b),
    io.in_a - io.in_b
  )
}
