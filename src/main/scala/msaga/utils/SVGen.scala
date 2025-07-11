package msaga.utils

import circt.stage._
import msaga.Configs


object SVGen {
  def main(args: Array[String]): Unit = {
    val rows = 128
    val cols = 128
    implicit val ev = Configs.fp16MulFp32AddArithmeticImpl
    ChiselStage.emitSystemVerilogFile(
      new msaga.sa.SystolicArray(rows, cols),
      args
    )
  }
}

