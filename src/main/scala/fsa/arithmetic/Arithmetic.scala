package fsa.arithmetic

import chisel3._

trait Arithmetic[T] {
  implicit def cast(self: T): ArithmeticOps[T]
}

trait ArithmeticOps[T] {
  def zero: T
  def one: T
  def minimum: T
  // log2(e) / sqrt(dk)
  def attentionScale(dk: Int): T
  def typeRepr: String
}

object Arithmetic {
  implicit object FPArithmetic extends Arithmetic[FloatPoint] {
    import easyfloat.{IEEEFloat, PyFPConst}
    override implicit def cast(self: FloatPoint): ArithmeticOps[FloatPoint] = new ArithmeticOps[FloatPoint] {
      override def zero = 0.U.asTypeOf(self)
      override def one = {
        val bits = IEEEFloat.expBias(self.expWidth) << self.mantissaWidth
        bits.U.asTypeOf(self)
      }
      override def minimum = {
        // -inf
        val sign = BigInt(1) << (self.expWidth + self.mantissaWidth)
        val exp = ((BigInt(1) << self.expWidth) - 1) << self.mantissaWidth
        val bits = sign | exp
        bits.U.asTypeOf(self)
      }
      override def attentionScale(dk: Int) = PyFPConst.attentionScale(
        self.expWidth, self.mantissaWidth,
        dk = dk,
        projectDir = "generators/easyfloat"
      ).U.asTypeOf(self)

      override def typeRepr: String = (self.expWidth, self.mantissaWidth) match {
        case (5, 10) => "fp16"
        case (8, 7) => "bf16"
        case (8, 23) => "fp32"
        case (e, m) => f"e${e}m$m"
      }
    }
  }
}

// import this to directly access T.zero / T.minimum
object ArithmeticSyntax {
  implicit class ArithmeticOpsSyntax[T](self: T)(implicit arith: Arithmetic[T]) {
    import arith._
    def zero: T = self.zero
    def one: T = self.one
    def minimum: T = self.minimum
    def attentionScale(dk: Int): T = self.attentionScale(dk)
    def typeRepr: String = self.typeRepr
  }
}


object MacCMD {
  def width = 1
  def MAC = 0.U(width.W)
  def EXP2 = 1.U(width.W)
}

object CmpCMD {
  def width = 1
  def MAX = 0.U(width.W)
  def SUB = 1.U(width.W)
}

abstract class MacUnit[E <: Data : Arithmetic, A <: Data : Arithmetic](val elemType: E, val accType: A) extends Module {
  val io = IO(new Bundle {
    val in_a = Input(elemType) // reg in PE
    val in_b = Input(elemType) // left input
    val in_c = Input(accType) // up/down input
    val in_cmd = Input(UInt(MacCMD.width.W))
    val out_accType = Output(accType)
    val out_elemType = Output(elemType)
    val out_exp2 = Output(Bool())
  })
}

abstract class CmpUnit[A <: Data](val accType: A) extends Module {
  val io = IO(new Bundle {
    val in_a = Input(accType)
    val in_b = Input(accType)
    val out_max = Output(accType)
    val out_diff = Output(accType)
  })
}

trait HasMultiCycleIO { this: Module =>
  val multiCycleIO = IO(new Bundle {
    val reciprocal_in_valid = Input(Bool())
    val reciprocal_out_valid = Output(Bool())
  })
}

trait HasArithmeticParams {
  val reciprocalLatency: Int
  val exp2PwlPieces: Int
}

abstract class ArithmeticImpl[E <: Data : Arithmetic, A <: Data : Arithmetic] extends HasArithmeticParams {
  def elemType: E
  def accType: A
  def peMac: MacUnit[E, A]
  def accUnit: MacUnit[A, A] with HasMultiCycleIO
  def accCmp: CmpUnit[A]
  def exp2PwlIntercepts: Seq[A]
  def exp2PwlSlopes: Seq[E]
  def viewAasE: A => E
  def viewEasA: E => A
  def cvtAtoE: A => E
}