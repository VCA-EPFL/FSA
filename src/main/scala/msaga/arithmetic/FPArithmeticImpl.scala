package msaga.arithmetic

import chisel3._
import easyfloat._
import ArithmeticSyntax._

final class FloatPoint(val expWidth: Int, val mantissaWidth: Int) extends Bundle {
  val sign = Bool()
  val exp = UInt(expWidth.W)
  val mantissa = UInt(mantissaWidth.W)
}

object FloatPoint {
  def apply(ew: Int, mw: Int): FloatPoint = new FloatPoint(ew, mw)
}

class FPMacUnit(mulEW: Int, mulMW: Int, addEW: Int, addMW: Int, pwlPieces: Int = 8) extends
  MacUnit[FloatPoint, FloatPoint](FloatPoint(mulEW, mulMW), FloatPoint(addEW, addMW))
{
  // TODO: avoid hard code projectDir
  val slopes = PyFPConst.slopes(mulEW, mulMW, projectDir = "generators/easyfloat", pieces = pwlPieces)
  val intercepts = PyFPConst.intercepts(mulEW, mulMW, projectDir = "generators/easyfloat", pieces = pwlPieces)
  val pwlConst = slopes.zip(intercepts)
  val mulAddExp2 = Module(new RawFloat_MulAddExp2(
    1 + mulEW, 1 + mulMW, 1 + addEW, 1 + addMW, pwlConst
  ))
  mulAddExp2.io.in_a.fromIEEE(io.in_a.asUInt, mulEW, mulMW)
  mulAddExp2.io.in_b.fromIEEE(io.in_b.asUInt, mulEW, mulMW)
  mulAddExp2.io.in_c.fromIEEE(io.in_c.asUInt, addEW, addMW)
  mulAddExp2.io.in_exp2 := io.in_cmd === MacCMD.EXP2
  io.out := Rounding.round(mulAddExp2.io.out, RoundingMode.RNE, addEW, addMW).asTypeOf(accType)
}

class FPCmpUnit(ew: Int, mw: Int) extends CmpUnit(FloatPoint(ew, mw)) {
  val fma = Module(new FMA(ew, mw, ew, mw))
  val negB = WireInit(io.in_b)
  negB.sign := !io.in_b.sign
  // always compute a - b = a * 1.0 + -b
  fma.io.a := io.in_a.asUInt
  fma.io.b := accType.one.asUInt
  fma.io.c := negB.asUInt
  val max = Mux(fma.io.out.asTypeOf(accType).sign,
    io.in_b, // a - b < 0, max = b
    io.in_a  // a - b > 0, max = a
  )
  io.out := Mux(io.in_cmd === CmpCMD.MAX, max, fma.io.out.asTypeOf(accType))
}

class FPArithmeticImpl(mulEW: Int, mulMW: Int, addEW: Int, addMW: Int)
  extends ArithmeticImpl[FloatPoint, FloatPoint]
{

  override def elemType: FloatPoint = FloatPoint(mulEW, mulMW)

  override def accType: FloatPoint = FloatPoint(addEW, addMW)

  override def peMac: MacUnit[FloatPoint, FloatPoint] = new FPMacUnit(mulEW, mulMW, addEW, addMW)

  override def accMac: MacUnit[FloatPoint, FloatPoint] = new FPMacUnit(addEW, addMW, addEW, addMW)

  override def accCmp: CmpUnit[FloatPoint] = new FPCmpUnit(addEW, addMW)
}
