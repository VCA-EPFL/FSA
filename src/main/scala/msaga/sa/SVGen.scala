package msaga.sa

import chisel3._
import circt.stage.ChiselStage

case class SAGenArgs(
                      dim: Int, elemWidth: Int, accWidth: Int,
                      chiselArgs: List[String] = Nil
                    )

object SVGen {

  def genSIntSystolicArray(dim: Int, elemWidth: Int, accWidth: Int) = {
    new SystolicArray(
      dim, dim, SInt(elemWidth.W), SInt(accWidth.W),
      () => new SIntMacUnit(elemWidth, accWidth),
      () => new SIntCmpUnit(accWidth)
    )
  }

  def parseArgs(args: Array[String]): SAGenArgs = {
    def nextOpt(cur: SAGenArgs, list: List[String]): SAGenArgs = {
      list match {
        case Nil => cur
        case "--dim" :: n :: tail =>
          nextOpt(cur.copy(dim = n.toInt), tail)
        case "--elem-width" :: n :: tail =>
          nextOpt(cur.copy(elemWidth = n.toInt), tail)
        case "--acc-width" :: n :: tail =>
          nextOpt(cur.copy(accWidth = n.toInt), tail)
        case unknown :: tail =>
          nextOpt(cur.copy(chiselArgs = cur.chiselArgs :+ unknown), tail)
      }
    }
    nextOpt(SAGenArgs(3, 16, 16), args.toList)
  }

  def main(args: Array[String]) {
    val genArgs = parseArgs(args)
    ChiselStage.emitSystemVerilogFile(
      genSIntSystolicArray(
        genArgs.dim, genArgs.elemWidth, genArgs.accWidth
      ),
      genArgs.chiselArgs.toArray
    )
  }
}