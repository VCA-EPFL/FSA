package msaga.svgen

import chisel3._
import circt.stage.ChiselStage
import msaga.sa.{SIntCmpUnit, SIntMacUnit, SystolicArray}

object SystolicArrayGen {

  def genSIntSystolicArray(dim: Int, elemWidth: Int, accWidth: Int) = {
    new SystolicArray(
      dim, dim, SInt(elemWidth.W), SInt(accWidth.W),
      () => new SIntMacUnit(elemWidth, accWidth),
      () => new SIntCmpUnit(accWidth)
    )
  }

  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    ChiselStage.emitSystemVerilogFile(
      genSIntSystolicArray(
        genArgs.dim, genArgs.elemWidth, genArgs.accWidth
      ),
      genArgs.chiselArgs.toArray
    )
  }
}