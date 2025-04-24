package msaga.svgen

import chisel3._
import circt.stage._
import chisel3.util._
import chisel3.stage.ChiselGeneratorAnnotation
import msaga.sa.{SIntCmpUnit, SIntMacUnit}
import msaga.{MSAGA, MSAGAKey, MSAGAParams}
import org.chipsalliance.cde.config.Config

object MSAGAGen {
  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    val rowBytes = genArgs.dim * genArgs.elemWidth / 8
    val rows =  3 * genArgs.dim // Q, K, V
    val params = MSAGAParams(genArgs.dim, rowBytes * rows, 0)
    implicit val config: Config = new Config((_, _, _) => {
      case MSAGAKey => params
    })

    ChiselStage.emitSystemVerilogFile(new MSAGA(
      SInt(genArgs.elemWidth.W), SInt(genArgs.accWidth.W),
      () => new SIntMacUnit(genArgs.elemWidth, genArgs.accWidth),
      () => new SIntCmpUnit(genArgs.elemWidth)
    ), genArgs.chiselArgs.toArray)
  }
}


