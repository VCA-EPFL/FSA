package msaga.svgen

import circt.stage._
import msaga.arithmetic.FPArithmeticImpl
import msaga.{MSAGA, MSAGAKey, MSAGAParams}
import org.chipsalliance.cde.config.Config

object MSAGAGen {
  def main(args: Array[String]): Unit = {
    val genArgs = GenOptions.parseArgs(args)
    val spRowBytes = genArgs.dim * genArgs.mulWidth / 8
    val accRowBytes = genArgs.dim * genArgs.addWidth / 8
    val params = MSAGAParams(genArgs.dim, spRowBytes * genArgs.spRows, accRowBytes * genArgs.accRows)
    implicit val config: Config = new Config((_, _, _) => {
      case MSAGAKey => params
    })

    ChiselStage.emitSystemVerilogFile(
      new MSAGA(new FPArithmeticImpl(
        genArgs.mulEW, genArgs.mulMW,
        genArgs.addEW, genArgs.addMW
      )),
      genArgs.chiselArgs.toArray
    )
  }
}


