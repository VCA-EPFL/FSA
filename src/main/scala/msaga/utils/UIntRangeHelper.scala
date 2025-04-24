package msaga.utils

import chisel3._

object UIntRangeHelper {
  implicit class UIntRangeHelper(x: UInt) {
    def between(start_inclusive: Int, end_exclusive: Int): Bool = start_inclusive.U <= x && x < end_exclusive.U
    def at(y: Int): Bool = x === y.U
  }
}
