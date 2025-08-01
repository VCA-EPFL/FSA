package fsa.arithmetic

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

class FPMacUnit(mulEW: Int, mulMW: Int, addEW: Int, addMW: Int, pwlPieces: Int) extends
  MacUnit[FloatPoint, FloatPoint](FloatPoint(mulEW, mulMW), FloatPoint(addEW, addMW))
{
  val mulAddExp2 = Module(new RawFloat_MulAddExp2(
    1 + mulEW, 1 + mulMW, 1 + addEW, 1 + addMW, Right(pwlPieces)
  ))
  mulAddExp2.io.in_a.fromIEEE(io.in_a.asUInt, mulEW, mulMW)
  mulAddExp2.io.in_b.fromIEEE(io.in_b.asUInt, mulEW, mulMW)
  mulAddExp2.io.in_c.fromIEEE(io.in_c.asUInt, addEW, addMW)
  mulAddExp2.io.in_exp2 := io.in_cmd === MacCMD.EXP2
  when(mulAddExp2.io.in_exp2) {
    // restore the correct exponent for in_c
    val c = WireInit(io.in_c)
    // set exp[MSB:1] to exp_bias[MSB:1]
    c.exp := Fill(addEW - 2, true.B) ## io.in_c.exp(0)
    mulAddExp2.io.in_c.fromIEEE(c.asUInt, addEW, addMW)
  }
  val pwlIndex = (io.in_c.exp >> 1.U).take(log2Up(pwlPieces))

  io.out_accType := Rounding.round(mulAddExp2.io.out, RoundingMode.RNE, addEW, addMW).asTypeOf(accType)
  io.out_elemType := Rounding.round(mulAddExp2.io.out, RoundingMode.RNE, mulEW, mulMW).asTypeOf(elemType)
  io.out_exp2 := mulAddExp2.io.in_exp2 && pwlIndex === mulAddExp2.io.exp2_frac_msb
}



class FPAccUnit(addEW: Int, addMW: Int, pwlPieces: Int, divBitsPerCycle: Int) extends
  MacUnit[FloatPoint, FloatPoint](FloatPoint(addEW, addMW), FloatPoint(addEW, addMW)) with HasMultiCycleIO
{
  // for accumulator, we use in-place pwl constants
  // TODO: avoid hard code projectDir
  val slopes = PyFPConst.slopes(addEW, addMW, projectDir = "generators/easyfloat", pieces = pwlPieces)
  val intercepts = PyFPConst.intercepts(addEW, addMW, projectDir = "generators/easyfloat", pieces = pwlPieces)
  val pwlConst = slopes.zip(intercepts)
  val mulAddExp2 = Module(new RawFloat_MulAddExp2(
    1 + addEW, 1 + addMW, 1 + addEW, 1 + addMW, Left(pwlConst)
  ))
  mulAddExp2.io.in_a.fromIEEE(io.in_a.asUInt, addEW, addMW)
  mulAddExp2.io.in_b.fromIEEE(io.in_b.asUInt, addEW, addMW)
  mulAddExp2.io.in_c.fromIEEE(io.in_c.asUInt, addEW, addMW)
  mulAddExp2.io.in_exp2 := io.in_cmd === MacCMD.EXP2

  // calculate reciprocal using res = 1.0 / a
  val div = Module(new RawFloat_Div(1 + addEW, 1 + addMW, divBitsPerCycle))
  div.io.in.valid := multiCycleIO.reciprocal_in_valid
  div.io.in.bits.a.fromIEEE(accType.one.asUInt, addEW, addMW)
  div.io.in.bits.b := mulAddExp2.io.in_a
  div.io.out.ready := true.B
  multiCycleIO.reciprocal_out_valid := div.io.out.valid

  require(div.io.out.bits.mantissa.getWidth == mulAddExp2.io.out.mantissa.getWidth)
  require(div.io.out.bits.exp.getWidth <= mulAddExp2.io.out.exp.getWidth)
  val rawOut = WireInit(mulAddExp2.io.out)
  // the output exp of mulAddExp2 is wider than the output of div, so we need to adjust it
  when(multiCycleIO.reciprocal_out_valid) {
    rawOut.sign := div.io.out.bits.sign
    // auto-sign-extend the exponent
    rawOut.exp := div.io.out.bits.exp
    rawOut.mantissa := div.io.out.bits.mantissa
    rawOut.isInf := div.io.out.bits.isInf
    rawOut.isNaN := div.io.out.bits.isNaN
    rawOut.isZero := div.io.out.bits.isZero
  }

  io.out_accType := Rounding.round(rawOut, RoundingMode.RNE, addEW, addMW).asTypeOf(accType)
  io.out_elemType := io.out_accType
  io.out_exp2 := mulAddExp2.io.in_exp2
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

class FPArithmeticImpl(mulEW: Int, mulMW: Int, addEW: Int, addMW: Int, pwlPieces: Int = 8, divBitsPerCycle: Int = 2)
  extends ArithmeticImpl[FloatPoint, FloatPoint]
{

  require(mulEW <= addEW && mulMW <= addMW)

  val isMixedPrecision = !(mulEW == addEW && mulMW == addMW)

  override def elemType: FloatPoint = FloatPoint(mulEW, mulMW)

  override def accType: FloatPoint = FloatPoint(addEW, addMW)

  override def peMac: MacUnit[FloatPoint, FloatPoint] = new FPMacUnit(mulEW, mulMW, addEW, addMW, pwlPieces)

  override def accUnit: MacUnit[FloatPoint, FloatPoint] with HasMultiCycleIO = new FPAccUnit(addEW, addMW, pwlPieces, divBitsPerCycle)

  override def accCmp: CmpUnit[FloatPoint] = new FPCmpUnit(addEW, addMW)

  override val reciprocalLatency: Int = Div.nCycles(addMW, divBitsPerCycle)

  override val exp2PwlPieces: Int = pwlPieces


  /*
    Slopes and intercepts for exp2 piecewise linear approximation.
    The slopes and intercepts generated by python are ordered by increasing x.
    In our PE, we use the exponent MSBs to select the piece,
    since x is in range (-1, 0], a larger exponent means smaller value,
    we reverse the order to get decreasing values.

    We have to attach the piece index information with the intercepts for
    PEs to select the right intercept.

    Notice that all intercepts are in the range of (0.5, 1], their exponent
    must be either 0 or -1, in the IEEE format, it would be exp_bias or exp_bias - 1,
    where only the lsb can be different, so we can safely place the counter in the
    exponent field.

    Here we replace exponent[MSB:1] with the counter value and only keep the lsb.
    Inside the PE, we can simply use `exp_bias[MSB:1] ## exp[0]` to get the exponent back.
  */
  override def exp2PwlIntercepts: Seq[FloatPoint] = {
    require(addEW - 1 >= log2Up(pwlPieces))
    val rawIntercepts = PyFPConst.intercepts(
      addEW, addMW,
      projectDir = "generators/easyfloat", pieces = pwlPieces
    ).reverse
    rawIntercepts.zipWithIndex.map{ case (raw, i) =>
      val rawFp = WireInit(raw.U((1 + addEW + addMW).W).asTypeOf(FloatPoint(addEW, addMW)))
      val fp = WireInit(rawFp)
      fp.exp := i.U ## rawFp.exp(0)
      fp
    }
  }

  override def exp2PwlSlopes: Seq[FloatPoint] = PyFPConst.slopes(
    mulEW, mulMW,
    projectDir = "generators/easyfloat", pieces = pwlPieces
  ).map(_.U((1 + mulEW + mulMW).W).asTypeOf(FloatPoint(mulEW, mulMW))).reverse

  override def viewAasE: FloatPoint => FloatPoint = { (a: FloatPoint) =>
    if (isMixedPrecision) {
      a.asUInt.take(elemType.getWidth).asTypeOf(elemType)
    } else a
  }

  override def viewEasA: FloatPoint => FloatPoint = { (e: FloatPoint) =>
    if (isMixedPrecision) {
      (0.U((accType.getWidth - elemType.getWidth).W) ## e.asUInt).asTypeOf(accType)
    } else e
  }

  override def cvtAtoE: FloatPoint => FloatPoint = { (a: FloatPoint) =>
    if (isMixedPrecision) {
      val rawA = Wire(new RawFloat(1 + addEW, 1 + addMW))
      rawA.fromIEEE(a.asUInt, addEW, addMW)
      Rounding.round(rawA, RoundingMode.RNE, mulEW, mulMW).asTypeOf(elemType)
    } else {
      a
    }
  }
}
