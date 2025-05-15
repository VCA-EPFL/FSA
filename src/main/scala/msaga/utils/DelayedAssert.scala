package msaga.utils

import chisel3._
import chisel3.experimental.SourceInfo

object DelayedAssert {
  // delay the `cond` to get more waveforms after the error occurs
  def apply(cond: Bool, delay: Int = 2)(implicit sourceInfo: SourceInfo): assert.Assert = {
    assert((0 until delay).foldLeft(cond){(c, _) => RegNext(c, init = true.B)})
  }
}
