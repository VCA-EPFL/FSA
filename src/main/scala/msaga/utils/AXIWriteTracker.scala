package msaga.utils

import chisel3._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.amba.axi4.AXI4BundleParameters
import chisel3.util.HasBlackBoxResource


class AXI4WriteTracker(params: AXI4BundleParameters) extends BlackBox(
  Map(
    "ADDR_BITS" -> params.addrBits,
    "SIZE_BITS" -> params.sizeBits,
    "LEN_BITS"  -> params.lenBits,
    "DATA_BITS" -> params.dataBits
  )
) with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val aw_fire = Input(Bool())
    val aw_addr = Input(UInt(params.addrBits.W))
    val aw_size = Input(UInt(params.sizeBits.W))
    val aw_len = Input(UInt(params.lenBits.W))
    val w_fire = Input(Bool())
    val w_data = Input(UInt(params.dataBits.W))
    val w_last = Input(Bool())
  })

  addResource("/msaga/vsrc/AXI4WriteTracker.v")
  addResource("/msaga/csrc/AXI4WriteTracker.cc")
}