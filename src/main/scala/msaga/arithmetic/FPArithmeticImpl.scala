package msaga.arithmetic

import chisel3._
import chisel3.util._
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
  val intercepts = PyFPConst.intercepts(addEW, addMW, projectDir = "generators/easyfloat", pieces = pwlPieces)
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

class FPAccUnit(addEW: Int, addMW: Int, pwlPieces: Int = 8) extends
  FPMacUnit(addEW, addMW, addEW, addMW, pwlPieces) with HasMultiCycleIO
{
  val reciprocal = Module(new Reciprocal(addEW, addMW))
  reciprocal.io.in := io.in_a.asUInt
  reciprocal.io.in_valid := multiCycleIO.reciprocal_in_valid
  reciprocal.io.fma_rounded_result := io.out.asUInt
  mulAddExp2.io.in_exp2 := !reciprocal.io.in_valid && io.in_cmd === MacCMD.EXP2
  when(reciprocal.io.in_valid) {
    mulAddExp2.io.in_a := reciprocal.io.fma_rawA
    mulAddExp2.io.in_b := reciprocal.io.fma_rawB
    mulAddExp2.io.in_c := reciprocal.io.fma_rawC
  }
  multiCycleIO.reciprocal_out_valid := reciprocal.io.out.valid
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
  io.out_max := max
  io.out_diff := fma.io.out.asTypeOf(accType)
}

class FPArithmeticImpl(mulEW: Int, mulMW: Int, addEW: Int, addMW: Int)
  extends ArithmeticImpl[FloatPoint, FloatPoint]
{

  override def elemType: FloatPoint = FloatPoint(mulEW, mulMW)

  override def accType: FloatPoint = FloatPoint(addEW, addMW)

  override def peMac: MacUnit[FloatPoint, FloatPoint] = new FPMacUnit(mulEW, mulMW, addEW, addMW)

  override def accUnit: MacUnit[FloatPoint, FloatPoint] with HasMultiCycleIO = new FPAccUnit(addEW, addMW)

  override def accCmp: CmpUnit[FloatPoint] = new FPCmpUnit(addEW, addMW)

  override val reciprocalLatency: Int = Reciprocal.nCycles(addMW)
}
