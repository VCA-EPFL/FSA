package msaga.arithmetic

import chisel3._

class SIntMacUnit(elemWidth: Int, accWidth: Int) extends
  MacUnit[SInt, SInt](SInt(elemWidth.W), SInt(accWidth.W))
{
  io.out := Mux(io.in_cmd === MacCMD.MAC,
    (io.in_a * io.in_b) +& io.in_c,
    io.in_a +& 1.S // FIXME: for testing only
  )
}

class SIntCmpUnit(width: Int) extends CmpUnit(SInt(width.W)) {
  io.out_max := Mux(io.in_a > io.in_b, io.in_a, io.in_b)
  io.out_diff := io.in_a - io.in_b
}

class SIntDummyImpl(ew: Int, aw: Int) extends ArithmeticImpl[SInt, SInt]{
  override def elemType: SInt = SInt(ew.W)

  override def accType: SInt = SInt(aw.W)

  override def peMac: MacUnit[SInt, SInt] = new SIntMacUnit(ew, aw)

  override def accUnit: MacUnit[SInt, SInt] with HasMultiCycleIO = new SIntMacUnit(aw, aw) with HasMultiCycleIO {
    multiCycleIO <> DontCare
  }

  override def accCmp: CmpUnit[SInt] = new SIntCmpUnit(aw)

  override val reciprocalLatency: Int = 1
}
