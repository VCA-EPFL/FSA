package fsa

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config._
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy.AddressSet
import freechips.rocketchip.regmapper.RegField
import fsa.frontend.{Decoder, Semaphores}
import fsa.arithmetic._
import fsa.dma.DMA
import fsa.utils.Ehr
import fsa.arithmetic.ArithmeticSyntax._
import freechips.rocketchip.util.ElaborationArtefacts
import freechips.rocketchip.subsystem.ExtMem

class AXI4FSA[E <: Data : Arithmetic, A <: Data : Arithmetic](val ev: ArithmeticImpl[E, A])(implicit p: Parameters) extends LazyModule {
  val instBeatBytes = 4
  val instBeatBits = instBeatBytes * 8
  val fsaParams = p(FSA).get

  val configNode = AXI4RegisterNode(
    address = AddressSet(0x8000, 0xff)
  )
  val dma = LazyModule(new DMA(
    nPorts = fsaParams.nMemPorts,
    sramAddrWidth = fsaParams.sramAddrWidth,
    dmaLoadInflight = fsaParams.dmaLoadInflight,
    dmaStoreInflight = fsaParams.dmaStoreInflight,
    spadElemWidth = ev.elemType.getWidth,
    spadRowSize = fsaParams.saRows,
    accElemWidth = ev.accType.getWidth,
    accRowSize = fsaParams.saCols
  ))
  val memNode = dma.node

  lazy val module = new LazyModuleImp(this) {

    val memAddrWidth = memNode.out.map(_._2.bundle.addrBits).max

    val s_idle :: s_active :: s_done :: Nil = Enum(3)
    val state = RegInit(s_idle)
    val set_active = WireInit(false.B)
    val set_done = Wire(Bool())
    val rawInstQueue = Module(
      new Queue(UInt(instBeatBits.W), fsaParams.instructionQueueEntries, useSyncReadMem = false)
    )

    val firstInstFire = RegInit(false.B)

    val enqInstCnt = RegInit(0.U(32.W))
    val deqInstCnt = RegInit(0.U(32.W))

    when(rawInstQueue.io.enq.fire) {
      enqInstCnt := enqInstCnt + 1.U
    }
    when(rawInstQueue.io.deq.fire) {
      deqInstCnt := deqInstCnt + 1.U
    }

    val perfCounters = scala.collection.mutable.TreeMap[String, UInt]()

    def addPerfCounter(name: String): UInt = {
      val counter = RegInit(0.U(32.W))
      counter.suggestName(f"perfCnt_$name")
      perfCounters(name) = counter
      counter
    }

    val perfCntExecTime = addPerfCounter("execTime")
    val perfCntMxBubble = addPerfCounter("mxBubble")
    val perfCntMxActive = addPerfCounter("mxActive")
    val perfCntDMAActive = addPerfCounter("dmaActive")
    val perfCntRawInst = addPerfCounter("rawInst")
    val perfCntMxInst = addPerfCounter("mxInst")
    val perfCntDMAInst = addPerfCounter("dmaInst")
    val perfCntFence = addPerfCounter("fence")

    configNode.regmap(
      0x00 -> Seq(RegField.w(instBeatBits, rawInstQueue.io.enq)),
      0x04 -> Seq(RegField.w(32, set_active)),
      0x08 -> Seq(RegField.r(32, state)),
      0x0C -> Seq(RegField.r(32, perfCntExecTime)),
      0x10 -> Seq(RegField.r(32, perfCntMxBubble)),
      0x14 -> Seq(RegField.r(32, perfCntMxActive)),
      0x18 -> Seq(RegField.r(32, perfCntDMAActive)),
      0x1C -> Seq(RegField.r(32, perfCntRawInst)),
      0x20 -> Seq(RegField.r(32, perfCntMxInst)),
      0x24 -> Seq(RegField.r(32, perfCntDMAInst)),
      0x28 -> Seq(RegField.r(32, perfCntFence)),
      0x2c -> Seq(RegField.r(32, enqInstCnt)),
      0x30 -> Seq(RegField.r(32, deqInstCnt)),
    )

    switch(state) {
      is(s_idle) {
        when(set_active) {
          state := s_active
        }
      }
      is(s_active) {
        when(set_done) {
          state := s_done
        }
      }
      is(s_done) {
        when(set_active) {
          perfCounters.values.foreach(_ := 0.U) // reset all perf counters
          state := s_active
        }
      }
    }


    val decoder = Module(new Decoder(memAddrWidth))
    val semaphores = Module(new Semaphores(nRead = 3, nWrite = 3))
    val dmaBeatBytes = memNode.out.head._2.slave.beatBytes
    val fsa = Module(new FSA(ev, dmaBeatBytes))

    when(fsa.io.inst.fire) {
      firstInstFire := true.B
    }.elsewhen(set_done) {
      firstInstFire := false.B
    }

    val is_active = state === s_active
    decoder.io.in.valid := rawInstQueue.io.deq.valid && is_active
    decoder.io.in.bits := rawInstQueue.io.deq.bits
    rawInstQueue.io.deq.ready := decoder.io.in.ready && is_active

    val mxInst = Queue(decoder.io.outMx, entries = fsaParams.mxInflight, pipe = true)
    // DMA has its own load/store queues inside it
    val dmaInst = Queue(decoder.io.outDMA, pipe = true)

    val dmaDone = RegNext(!(dma.module.io.busy || dmaInst.valid), init = false.B)
    val mxDone = RegNext(!(fsa.io.busy || mxInst.valid), init = false.B)
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
    when(fsa.io.inst.fire) {
      mxAcqFlag.write(1, false.B)
    }
    val mxDepReady = !mxInst.bits.header.acquireValid || mxAcqFlag.read(1)

    fsa.io.inst.valid := mxInst.valid && mxDepReady
    fsa.io.inst.bits := mxInst.bits
    mxInst.ready := fsa.io.inst.ready && mxDepReady

    mxSemRelease <> fsa.io.sem_release

    fsa.io.spad_write <> dma.module.io.spadWrite
    fsa.io.acc_read <> dma.module.io.accRead

    dma.module.io.inst <> dmaInst

    semaphores.io.acquire.tail.zip(dma.module.io.semaphoreAcquire).foreach { case (acq, dmaAcq) =>
      acq <> dmaAcq
    }
    semaphores.io.release.tail.zip(dma.module.io.semaphoreRelease).foreach { case (rel, dmaRel) =>
      rel <> dmaRel
    }

    when(state === s_active) {
      perfCntExecTime := perfCntExecTime + 1.U
      when(firstInstFire) {
        when(fsa.io.inst.ready && !fsa.io.inst.valid) {
          perfCntMxBubble := perfCntMxBubble + 1.U
        }
      }
      when(fsa.io.busy) {
        perfCntMxActive := perfCntMxActive + 1.U
      }
      when(dma.module.io.active) {
        perfCntDMAActive := perfCntDMAActive + 1.U
      }
      when(rawInstQueue.io.deq.fire) {
        perfCntRawInst := perfCntRawInst + 1.U
      }
      when(fsa.io.inst.fire) {
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
      for ((name, counter) <- perfCounters) {
        printf(s"FSA: $name = %d\n", counter)
      }
    }


    val memParams = p(AXI4DirectMemPortKey).getOrElse(p(ExtMem).get)
    val configJSON = f"""
    |{
    |"sa_rows": ${fsaParams.saRows},
    |"sa_cols": ${fsaParams.saCols},
    |"inst_queue_size": ${fsaParams.instructionQueueEntries},
    |"e_type": "${ev.elemType.typeRepr}",
    |"a_type": "${ev.accType.typeRepr}",
    |"mem_base": ${memParams.master.base},
    |"mem_size": ${memParams.master.size},
    |"mem_align": ${memParams.master.beatBytes},
    |"spad_base": 0,
    |"spad_size": ${fsaParams.spadRows * fsaParams.saRows * ev.elemType.getWidth / 8},
    |"acc_base": 0,
    |"acc_size": ${fsaParams.accRows * fsaParams.saCols * ev.accType.getWidth / 8}
    |}
    """.stripMargin

    ElaborationArtefacts.add(
      "FSAConfig.json",
      configJSON
    )
  }
}

