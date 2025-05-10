package msaga.isa

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.UIntIsOneOf

import ISA.Constants._

trait HasInstructionType { this: Bundle =>
  val instType = UInt(I_TYPE_BITS.W)
}

trait HasAddr {this: Bundle =>
  def addrWidth: Int
  def maxAddrWidth: Int
  val _pad_addr_msb = if(maxAddrWidth > addrWidth) Some(UInt((maxAddrWidth - addrWidth).W)) else None
  val addr = UInt(addrWidth.W)
}

abstract class NBytesBundle(n: Int) extends Bundle {
  def checkWidth(): Unit = {
    require(this.getWidth == n * 8, f"width: ${this.getWidth} n: $n")
  }
  def padOpt(existingBits: Int) = if (existingBits == n * 8) None else Some(UInt((n * 8 - existingBits).W))
}

object ISA {

  object Constants {
    val I_TYPE_BITS = 3
    val N_SEMAPHORES = 32
    val SEM_ID_BITS = log2Up(N_SEMAPHORES)
    val SEM_VALUE_BITS = 3

    val MX_FUNC_BITS = 5
    val SPAD_MAX_ADDR_BITS = 20
    val SPAD_STRIDE_BITS = 5
    val ACC_MAX_ADDR_BITS = 20
    val ACC_STRIDE_BITS = 5
    val SRAM_MAX_ADDR_BITS = Seq(SPAD_MAX_ADDR_BITS, ACC_MAX_ADDR_BITS).max
    val SRAM_STRIDE_BITS = Seq(SPAD_STRIDE_BITS, ACC_STRIDE_BITS).max

    val DMA_FUNC_BITS = 4
    val DMA_SIZE_BITS = 10
    val DMA_REPEAT_BITS = 9
    val MEM_MAX_ADDR_BITS = 39
    val MEM_STRIDE_BITS = 10
  }

  object InstTypes {
    val FENCE = 0
    val MATRIX = 1
    val DMA = 2
  }

  object MxFunc {
    def LOAD_STATIONARY = 0.U
    def ATTENTION_SCORE_COMPUTE = 1.U
    def ATTENTION_VALUE_COMPUTE = 2.U
    def ATTENTION_LSE_NORM_SCALE = 3.U
    def ATTENTION_LSE_NORM = 4.U
  }

  object DMAFunc {
    def LD_SRAM = 0.U
    def ST_SRAM = 1.U
  }

}
