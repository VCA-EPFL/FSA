package msaga.frontend

import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{RegField, RegWriteFn}
import chisel3._
import chisel3.util._

class AXI4Frontend(implicit p: Parameters) extends LazyModule {
  val beatBytes = 4
  val beatBits = beatBytes * 8
  val configNode = AXI4RegisterNode(
    address = AddressSet(0x8000, 0xff)
  )
  lazy val module = new AXI4FrontendImpl(this)
}

class AXI4FrontendImpl(outer: AXI4Frontend) extends LazyModuleImp(outer) {

  val busy = RegInit(false.B)
  val entries = 256
  val rawInstQueue = Module(new Queue(UInt(outer.beatBits.W), entries, useSyncReadMem = true))

  outer.configNode.regmap(
    0x00 -> Seq(RegField.w(outer.beatBits, rawInstQueue.io.enq)),
    0x04 -> Seq(RegField.w(1, busy))
  )

  rawInstQueue.io.deq.ready := busy
  when(rawInstQueue.io.deq.fire) {
    printf(cf"inst: ${Hexadecimal(rawInstQueue.io.deq.bits)}\n")
  }

}
