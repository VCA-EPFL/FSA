package msaga.svgen

import circt.stage.ChiselStage
import chisel3._
import msaga.arithmetic.SIntDummyImpl
import msaga.sa.SystolicArray

object SystolicArrayGen {

  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    implicit val ev: SIntDummyImpl = new SIntDummyImpl(genArgs.elemWidth, genArgs.accWidth)
    ChiselStage.emitSystemVerilogFile(
      new SystolicArray[SInt, SInt](genArgs.dim, genArgs.dim),
      genArgs.chiselArgs.toArray
    )
  }
}