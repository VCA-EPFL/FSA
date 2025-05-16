package msaga

import chisel3._
import chisel3.util._

abstract class BaseSRAMIO(addrWidth: Int) extends Bundle {
  val valid = Output(Bool())
  val addr = Output(UInt(addrWidth.W))
  val ready = Input(Bool())
  def fire: Bool = valid && ready
}

class SRAMRead[T <: Data](addrWidth: Int, elem: T, n: Int) extends BaseSRAMIO(addrWidth) {
  val data = Input(Vec(n, elem))
}

class SRAMWrite[T <: Data](addrWidth: Int, elem: T, n: Int) extends BaseSRAMIO(addrWidth) {
  val mask = Output(Vec(n, Bool()))
  val data = Output(Vec(n, elem))
}

/**
 * A banked SRAM module for concurrent read and write operations.
 *
 * Conflict Resolution:
 * - If multiple read or write requests target the same address or bank,
 *   priority is always given to the request with the lowest index.
 * - Note: Conflicts between read and write requests are not handled by this module.
 *   It is the end user's responsibility to guarantee no Read-Write (RW) conflicts occur.
 *
 * @param rows Logical total sram rows
 * @param elem Element at the min granularity of masked read/write
 * @param n Number of elements to read/write per access
 * @param nBanks Number of physical banks
 */
class BankedSRAM[T <: Data]
(
  rows: Int, elem: T, n: Int,
  nBanks: Int,
  nReadPorts: Int, nWritePorts: Int
) extends Module {

  val addrWidth = log2Up(rows)
  val io = IO(new Bundle {
    val read = Vec(nReadPorts, Flipped(new SRAMRead(addrWidth, elem, n)))
    val write = Vec(nWritePorts, Flipped(new SRAMWrite(addrWidth, elem, n)))
  })

  require(isPow2(nBanks) && nBanks >= 2)

  val banks = (0 until nBanks).map(_ => SRAM.masked(
    size = rows / nBanks, tpe = Vec(n, elem),
    numReadPorts = 1, numWritePorts = 1, numReadwritePorts = 0
  ))

  def getBankIdx(addr: UInt): UInt = addr.take(log2Up(nBanks))
  def getBankAddr(addr: UInt): UInt = (addr >> log2Up(nBanks)).asUInt
  def readData(bankIdx: UInt): Vec[T] = {
    val bankIdxReg = RegNext(bankIdx)
    // perform mux at element-wise to reduce fanout
    VecInit(banks.map(_.readPorts.head.data).transpose.map(elemGroup => VecInit(elemGroup)(bankIdxReg)))
  }
  def writeData(grantMask: Vec[Bool]): Vec[T] = {
    VecInit(io.write.map(_.data).transpose.map(elemGroup => Mux1H(grantMask, elemGroup)))
  }

  // check conflict of bank i
  def check(ports: Seq[BaseSRAMIO], i: Int): (Bool, UInt, Vec[Bool]) = {
    val grantMask = Wire(Vec(ports.length, Bool()))
    val (en, addr) = ports.zip(grantMask).foldLeft(
      ( false.B, 0.U((addrWidth - log2Up(nBanks)).W) )
    ) { case ((prevGrant, prevAddr), (r, currGrant)) =>
      currGrant := !prevGrant && getBankIdx(r.addr) === i.U && r.valid
      val accGrant = prevGrant || currGrant
      val accAddr = Mux(currGrant, getBankAddr(r.addr), prevAddr)
      (accGrant, accAddr)
    }
    (en, addr, grantMask)
  }

  io.read.foreach(r => r.data := readData(getBankIdx(r.addr)))
  val (readGrantMask, writeGrantMask) = banks.zipWithIndex.map{ case (bank, i) =>
    val (r_en, r_addr, r_grant) = check(io.read, i)
    bank.readPorts.head.enable := r_en
    bank.readPorts.head.address := r_addr
    val (w_en, w_addr, w_grant) = check(io.write, i)
    bank.writePorts.head.enable := w_en
    bank.writePorts.head.address := w_addr
    bank.writePorts.head.mask.get := Mux1H(w_grant, io.write.map(_.mask))
    bank.writePorts.head.data := writeData(w_grant)
    (r_grant, w_grant)
  }.unzip
  readGrantMask.transpose.map(Cat(_).orR).zip(io.read).foreach{ case (g, r) => r.ready := g }
  writeGrantMask.transpose.map(Cat(_).orR).zip(io.write).foreach{ case (g, w) => w.ready := g }
}
