package msaga.svgen

import circt.stage._
import msaga.arithmetic.{FPArithmeticImpl, SIntDummyImpl}
import msaga.{MSAGA, MSAGAKey, MSAGAParams}
import org.chipsalliance.cde.config.Config

object MSAGAGen {
  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    val spRowBytes = genArgs.dim * genArgs.elemWidth / 8
    val spRows = 3 * genArgs.dim // Q, K, V
    val accRowBytes = genArgs.dim * genArgs.accWidth / 8
    val accRows = genArgs.dim + 1
    val params = MSAGAParams(genArgs.dim, spRowBytes * spRows, accRowBytes * accRows)
    implicit val config: Config = new Config((_, _, _) => {
      case MSAGAKey => params
    })

    ChiselStage.emitSystemVerilogFile(
      new MSAGA(new FPArithmeticImpl(8, 23, 8, 23)),
      genArgs.chiselArgs.toArray
    )
  }
}


