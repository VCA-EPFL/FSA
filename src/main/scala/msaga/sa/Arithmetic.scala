package msaga.sa

import chisel3._

/* Type class for numeric data types, similar to UCB/GEMMINI https://github.com/ucb-bar/gemmini.
   However, in our implementation the Arithmetic type class only requires `zero` and `min`,
   other complex operations like mac and cmp are separated out to enable:
   1. hardware resource sharing, e.g. the add/sub/mac/exp should share the same function unit.
   2. decoupled type definition and function unit implementation
*/

trait Arithmetic[T] {
  implicit def cast(self: T): ArithmeticOps[T]
}

trait ArithmeticOps[T] {
  def zero: T
  def one: T
  def minimum: T
  def lg2_e: T
}

object Arithmetic {
  implicit object SIntArithmetic extends Arithmetic[SInt] {
    override implicit def cast(self: SInt): ArithmeticOps[SInt] = new ArithmeticOps[SInt] {
      override def zero = 0.S
      override def one = 1.S
      override def minimum: SInt = (-(1 << (self.getWidth-1))).S
      // FIXME: this is for testing purpose only
      override def lg2_e = 2.S
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
    def lg2_e: T = self.lg2_e
  }
}