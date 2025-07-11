package fsa

import chisel3._
import chisel3.util._

trait HasSubBankParams {
  def rowSize: Int
  def elemWidth: Int
  def beatBytes: Int
  def nSubBanks: Int = {
    val n = rowSize * elemWidth / 8 / beatBytes
    require(n * beatBytes * 8 == rowSize * elemWidth)
    n
  }
}

abstract class BaseSRAMIO extends Bundle with HasSubBankParams {
  def addrWidth: Int

  val valid = Output(Bool())
  val addr = Output(UInt(addrWidth.W))
  val ready = Input(Bool())
  val data: Vec[UInt]

  def fire: Bool = valid && ready
  def dataSize: Int

  def bankValid(bankIdx: Int, nBanks: Int): Bool = {
    addr.take(log2Up(nBanks)) === bankIdx.U
  }
  def bankReady(subBankReady: Vec[Bool]): Bool
  def readBankData(in: Vec[Vec[UInt]]): Unit
  def subBankValid(idx: Int): Bool
  def subBankData(idx: Int): Vec[UInt]
}

trait HasFullRowAccess { this: BaseSRAMIO =>
  val subBankMask = Output(Vec(nSubBanks, Bool()))

  def setFullMask(): Unit = {
    subBankMask := (~0.U(nSubBanks.W)).asTypeOf(subBankMask)
  }

  override def dataSize: Int = rowSize
  override def subBankValid(idx: Int): Bool = subBankMask(idx)
  override def subBankData(idx: Int): Vec[UInt] = data.asTypeOf(
    Vec(nSubBanks, Vec(dataSize / nSubBanks, UInt(elemWidth.W)))
  )(idx)
  override def bankReady(subBankReady: Vec[Bool]): Bool = Cat(subBankMask.zip(subBankReady).map{
    case (m, r) =>
      !m || r
  }).andR
  override def readBankData(in: Vec[Vec[UInt]]): Unit = {
    data := in.asTypeOf(data)
  }
}

trait HasNarrowAccess { this: BaseSRAMIO =>
  val subBankIdx = Output(UInt(log2Up(nSubBanks).W))

  override def dataSize: Int = rowSize / nSubBanks
  override def subBankValid(idx: Int): Bool = subBankIdx === idx.U
  override def subBankData(idx: Int): Vec[UInt] = data
  override def bankReady(subBankReady: Vec[Bool]): Bool = subBankReady(subBankIdx)
  override def readBankData(in: Vec[Vec[UInt]]): Unit = {
    data := in(RegNext(subBankIdx))
  }
}

trait HasReadData { this: BaseSRAMIO =>
  override val data: Vec[UInt] = Input(Vec(dataSize, UInt(elemWidth.W)))
}

trait HasWriteData { this: BaseSRAMIO =>
  override val data: Vec[UInt] = Output(Vec(dataSize, UInt(elemWidth.W)))
}

class SRAMFullRead(val addrWidth: Int, val elemWidth: Int, val rowSize: Int, val beatBytes: Int)
  extends BaseSRAMIO with HasFullRowAccess with HasReadData

class SRAMFullWrite(val addrWidth: Int, val elemWidth: Int, val rowSize: Int, val beatBytes: Int)
  extends BaseSRAMIO with HasFullRowAccess with HasWriteData


class SRAMNarrowRead(val addrWidth: Int, val elemWidth: Int, val rowSize: Int, val beatBytes: Int)
  extends BaseSRAMIO with HasNarrowAccess with HasReadData

class SRAMNarrowWrite(val addrWidth: Int, val elemWidth: Int, val rowSize: Int, val beatBytes: Int)
  extends BaseSRAMIO with HasNarrowAccess with HasWriteData

/**
 * |       bank      |   |       bank      |
 * |sub bank|sub bank|   |sub bank|sub bank|
 * |--------|--------|   |--------|--------|
 * |--------|--------|   |--------|--------|
 */
class BankedSRAM
(
  rows: Int, val rowSize: Int, val elemWidth: Int,
  nBanks: Int, val beatBytes: Int,
  nFullRead: Int, nFullWrite: Int,
  nNarrowRead: Int, nNarrowWrite: Int,
  moduleName: String
) extends Module with HasSubBankParams {
  val addrWidth = log2Up(rows)

  val io = IO(new Bundle {
    val fullRead = Vec(nFullRead, Flipped(new SRAMFullRead(addrWidth, elemWidth, rowSize, beatBytes)))
    val fullWrite = Vec(nFullWrite, Flipped(new SRAMFullWrite(addrWidth, elemWidth, rowSize, beatBytes)))
    val narrowRead = Vec(nNarrowRead, Flipped(new SRAMNarrowRead(addrWidth, elemWidth, rowSize, beatBytes)))
    val narrowWrite = Vec(nNarrowWrite, Flipped(new SRAMNarrowWrite(addrWidth, elemWidth, rowSize, beatBytes)))
  })

  val banks: Seq[Seq[SRAMInterface[Vec[UInt]]]] = Seq.fill(nBanks) {
    val subBanks = Seq.fill(nSubBanks) {
      val sram = SRAM(
        rows, Vec(rowSize / nSubBanks, UInt(elemWidth.W)),
        numReadPorts = 1, numWritePorts = 1, numReadwritePorts = 0
      )
      sram
    }
    subBanks
  }

  def getBankIdx(req: BaseSRAMIO): UInt = req.addr.take(log2Up(nBanks))

  def check(reqs: Seq[BaseSRAMIO], bankIdx: Int, subBankIdx: Int): (Vec[Bool], Vec[Bool]) = {
    val readyMask = Wire(Vec(reqs.size, Bool()))
    val validMask = VecInit(reqs.map(r => r.valid && r.bankValid(bankIdx, nBanks) && r.subBankValid(subBankIdx)))
    validMask.zip(readyMask).foldLeft(false.B){ case (occupied, (v, ready)) =>
      ready := !occupied
      occupied || v
    }
    (validMask, readyMask)
  }

  val (bankReadReady, bankWriteReady) = banks.zipWithIndex.map{ case (bank, bankIdx)=>
    val (readReady, writeReady) = bank.zipWithIndex.map{ case (subBank, subBankIdx) =>
      val readReqs = io.fullRead ++ io.narrowRead
      val (readValid, readReady) = check(readReqs, bankIdx, subBankIdx)
      val readFire = readValid.zip(readReady).map{ case (v, r) => v && r }
      subBank.readPorts.head.enable := readValid.asUInt.orR
      subBank.readPorts.head.address := Mux1H(readFire, readReqs.map(_.addr))

      val writeReqs = io.fullWrite ++ io.narrowWrite
      val (writeValid, writeReady) = check(writeReqs, bankIdx, subBankIdx)
      val writeFire = writeValid.zip(writeReady).map{ case (v, r) => v && r }

      subBank.writePorts.head.enable := writeValid.asUInt.orR
      subBank.writePorts.head.address := Mux1H(writeFire, writeReqs.map(_.addr))
      subBank.writePorts.head.data := Mux1H(writeFire, writeReqs.map(_.subBankData(subBankIdx)))

      (readReady, writeReady)
    }.unzip

    val bankReadReady = (io.fullRead ++ io.narrowRead).zip(readReady.transpose).map { case (req, subBankReady) =>
      req.bankReady(VecInit(subBankReady))
    }
    val bankWriteReady = (io.fullWrite ++ io.narrowWrite).zip(writeReady.transpose).map { case (req, subBankReady) =>
      req.bankReady(VecInit(subBankReady))
    }
    (bankReadReady, bankWriteReady)
  }.unzip

  val bankReadData = VecInit(banks.map(b => VecInit(b.map(_.readPorts.head.data))))
  (io.fullRead ++ io.narrowRead).zip(bankReadReady.transpose).foreach{ case (req, bankReady) =>
    val bankIdx = getBankIdx(req)
    val bankIdxReg = RegNext(bankIdx)
    req.ready := VecInit(bankReady)(getBankIdx(req))
    req.readBankData(bankReadData(bankIdxReg))
  }
  (io.fullWrite ++ io.narrowWrite).zip(bankWriteReady.transpose).foreach{ case (req, bankReady) =>
    req.ready := VecInit(bankReady)(getBankIdx(req))
  }

  override def desiredName: String = moduleName
}