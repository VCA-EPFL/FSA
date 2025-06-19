package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy.AddressSet
import freechips.rocketchip.regmapper.RegField
import msaga.frontend.{Decoder, Semaphores}
import msaga.arithmetic._
import msaga.dma.DMA
import msaga.utils.Ehr
import msaga.arithmetic.ArithmeticSyntax._
import freechips.rocketchip.util.ElaborationArtefacts
import freechips.rocketchip.subsystem.ExtMem

class AXI4MSAGA[E <: Data : Arithmetic, A <: Data : Arithmetic](val ev: ArithmeticImpl[E, A])(implicit p: Parameters) extends LazyModule {
  val instBeatBytes = 4
  val instBeatBits = instBeatBytes * 8
  val msagaParams = p(MSAGAKey).get

  val configNode = AXI4RegisterNode(
    address = AddressSet(0x8000, 0xff)
  )
  val dma = LazyModule(new DMA(
    nPorts = msagaParams.nMemPorts,
    sramAddrWidth = msagaParams.sramAddrWidth,
    dmaLoadInflight = msagaParams.dmaLoadInflight,
    dmaStoreInflight = msagaParams.dmaStoreInflight,
    spadElemWidth = ev.elemType.getWidth,
    spadRowSize = msagaParams.saRows,
    accElemWidth = ev.accType.getWidth,
    accRowSize = msagaParams.saCols
  ))
  val memNode = dma.node

  lazy val module = new LazyModuleImp(this) {

    val memAddrWidth = memNode.out.map(_._2.bundle.addrBits).max

    val s_idle :: s_active :: s_done :: Nil = Enum(3)
    val state = RegInit(s_idle)
    val set_active = WireInit(false.B)
    val set_done = Wire(Bool())
    val rawInstQueue = Module(
      new Queue(UInt(instBeatBits.W), msagaParams.instructionQueueEntries, useSyncReadMem = true)
    )

    val firstInstFire = RegInit(false.B)

    val perfCntExecTime = RegInit(0.U(32.W))
    val perfCntMxWait = RegInit(0.U(32.W))
    val perfCntMxBusy = RegInit(0.U(32.W))
    val perfCntDMAWait = RegInit(0.U(32.W))
    val perfCntDMABusy = RegInit(0.U(32.W))
    val perfCntInstWait = RegInit(0.U(32.W))
    val perfCntInstBusy = RegInit(0.U(32.W))
    val perfCntRawInst = RegInit(0.U(32.W))
    val perfCntMxInst = RegInit(0.U(32.W))
    val perfCntDMAInst = RegInit(0.U(32.W))
    val perfCntFence = RegInit(0.U(32.W))


    configNode.regmap(
      0x00 -> Seq(RegField.w(instBeatBits, rawInstQueue.io.enq)),
      0x04 -> Seq(RegField.w(1, set_active)),
      0x08 -> Seq(RegField.r(2, state)),
      0x0C -> Seq(RegField.r(32, perfCntExecTime)),
      0x10 -> Seq(RegField.r(32, perfCntMxWait)),
      0x14 -> Seq(RegField.r(32, perfCntMxBusy)),
      0x18 -> Seq(RegField.r(32, perfCntDMAWait)),
      0x1C -> Seq(RegField.r(32, perfCntDMABusy)),
      0x20 -> Seq(RegField.r(32, perfCntInstWait)),
      0x24 -> Seq(RegField.r(32, perfCntInstBusy)),
      0x28 -> Seq(RegField.r(32, perfCntRawInst)),
      0x2C -> Seq(RegField.r(32, perfCntMxInst)),
      0x30 -> Seq(RegField.r(32, perfCntDMAInst)),
      0x34 -> Seq(RegField.r(32, perfCntFence))
    )

    switch(state) {
      is(s_idle) {
        when(set_active) {
          state := s_active
        }
      }
      is(s_active) {
        when(firstInstFire) {
          perfCntExecTime := perfCntExecTime + 1.U
        }
        when(set_done) {
          state := s_done
        }
      }
      is(s_done) {
        when(set_active) {
          perfCntExecTime := 0.U
          perfCntMxWait := 0.U
          perfCntMxBusy := 0.U
          perfCntDMAWait := 0.U
          perfCntDMABusy := 0.U
          perfCntInstWait := 0.U
          perfCntInstBusy := 0.U
          perfCntRawInst := 0.U
          perfCntMxInst := 0.U
          perfCntDMAInst := 0.U
          perfCntFence := 0.U
          state := s_active
        }
      }
    }


    val decoder = Module(new Decoder(memAddrWidth))
    val semaphores = Module(new Semaphores(nRead = 2, nWrite = 2))
    val dmaBeatBytes = memNode.out.head._2.slave.beatBytes
    val msaga = Module(new MSAGA(ev, dmaBeatBytes))

    when(msaga.io.inst.fire) {
      firstInstFire := true.B
    }.elsewhen(set_done) {
      firstInstFire := false.B
    }

    val is_active = state === s_active
    decoder.io.in.valid := rawInstQueue.io.deq.valid && is_active
    decoder.io.in.bits := rawInstQueue.io.deq.bits
    rawInstQueue.io.deq.ready := decoder.io.in.ready && is_active

    val mxInst = Queue(decoder.io.outMx, entries = msagaParams.mxInflight, pipe = true)
    // DMA has its own load/store queues inside it
    val dmaInst = Queue(decoder.io.outDMA, pipe = true)

    val dmaDone = RegNext(!(dma.module.io.busy || dmaInst.valid), init = false.B)
    val mxDone = RegNext(!(msaga.io.busy || mxInst.valid), init = false.B)
    val fenceReady = (!decoder.io.outFence.bits.dma || dmaDone) &&
      (!decoder.io.outFence.bits.dma || mxDone)
    decoder.io.outFence.ready := fenceReady
    set_done := decoder.io.outFence.fire && decoder.io.outFence.bits.stop


    val mxSemAcquire = semaphores.io.acquire.head
    val mxAcqFlag = Ehr(2, Bool(), Some(false.B))
    val mxSemRelease = semaphores.io.release.head
    mxSemAcquire.valid := mxInst.bits.header.acquireValid && mxInst.valid
    mxSemAcquire.bits.id := mxInst.bits.header.semId
    mxSemAcquire.bits.value := mxInst.bits.header.acquireSemValue
    when(mxSemAcquire.fire){
      mxAcqFlag.write(0, true.B)
    }
    when(msaga.io.inst.fire) {
      mxAcqFlag.write(1, false.B)
    }
    val mxDepReady = !mxInst.bits.header.acquireValid || mxAcqFlag.read(1)

    msaga.io.inst.valid := mxInst.valid && mxDepReady
    msaga.io.inst.bits := mxInst.bits
    mxInst.ready := msaga.io.inst.ready && mxDepReady

    mxSemRelease <> msaga.io.sem_release

    msaga.io.spad_write <> dma.module.io.spadWrite
    msaga.io.acc_read <> dma.module.io.accRead

    dma.module.io.inst <> dmaInst

    val dmaSemAcquire = semaphores.io.acquire.last
    val dmaSemRelease = semaphores.io.release.last

    dmaSemAcquire <> dma.module.io.semaphoreAcquire
    dmaSemRelease <> dma.module.io.semaphoreRelease

    when(state === s_active) {
      when(mxInst.valid && !mxDepReady) {
        perfCntMxWait := perfCntMxWait + 1.U
      }
      when(msaga.io.busy) {
        perfCntMxBusy := perfCntMxBusy + 1.U
      }
      when(dmaInst.valid && !dma.module.io.busy) {
        perfCntDMAWait := perfCntDMAWait + 1.U
      }
      when(dma.module.io.busy) {
        perfCntDMABusy := perfCntDMABusy + 1.U
      }
      when(rawInstQueue.io.deq.valid && !rawInstQueue.io.deq.ready) {
        perfCntInstBusy := perfCntInstBusy + 1.U
      }
      when(!rawInstQueue.io.deq.valid && rawInstQueue.io.deq.ready && firstInstFire) {
        perfCntInstWait := perfCntInstWait + 1.U
      }
      when(rawInstQueue.io.deq.fire) {
        perfCntRawInst := perfCntRawInst + 1.U
      }
      when(msaga.io.inst.fire) {
        perfCntMxInst := perfCntMxInst + 1.U
      }
      when(dma.module.io.inst.fire) {
        perfCntDMAInst := perfCntDMAInst + 1.U
      }
      when(decoder.io.outFence.fire) {
        perfCntFence := perfCntFence + 1.U
      }
    }

    when(RegNext(set_done, false.B)) {
      printf("MSAGA: exec time %d, mx wait %d, mx busy %d, dma wait %d, dma busy %d, inst wait %d, inst busy %d\n",
        perfCntExecTime,
        perfCntMxWait, perfCntMxBusy,
        perfCntDMAWait, perfCntDMABusy,
        perfCntInstWait, perfCntInstBusy
      )
    }


    val memParams = p(AXI4DirectMemPortKey).getOrElse(p(ExtMem).get)
    val configJSON = f"""
    |{
    |"sa_rows": ${msagaParams.saRows},
    |"sa_cols": ${msagaParams.saCols},
    |"inst_queue_size": ${msagaParams.instructionQueueEntries},
    |"e_type": "${ev.elemType.typeRepr}",
    |"a_type": "${ev.accType.typeRepr}",
    |"mem_base": ${memParams.master.base},
    |"mem_size": ${memParams.master.size},
    |"mem_align": ${memParams.master.beatBytes},
    |"spad_base": 0,
    |"spad_size": ${msagaParams.spadRows * msagaParams.saRows * ev.elemType.getWidth / 8},
    |"acc_base": 0,
    |"acc_size": ${msagaParams.accRows * msagaParams.saCols * ev.accType.getWidth / 8}
    |}
    """.stripMargin

    ElaborationArtefacts.add(
      "MSAGAConfig.json",
      configJSON
    )
  }
}

