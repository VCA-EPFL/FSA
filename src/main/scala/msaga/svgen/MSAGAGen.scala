package msaga.svgen

import circt.stage._
import msaga.arithmetic.SIntDummyImpl
import msaga.{MSAGA, MSAGAKey, MSAGAParams}
import org.chipsalliance.cde.config.Config

object MSAGAGen {
  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    val rowBytes = genArgs.dim * genArgs.elemWidth / 8
    val rows = 3 * genArgs.dim // Q, K, V
    val params = MSAGAParams(genArgs.dim, rowBytes * rows, 0)
    implicit val config: Config = new Config((_, _, _) => {
      case MSAGAKey => params
    })

    ChiselStage.emitSystemVerilogFile(
      new MSAGA(new SIntDummyImpl(genArgs.elemWidth, genArgs.accWidth)),
      genArgs.chiselArgs.toArray
    )
  }
}


