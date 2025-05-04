package msaga.svgen

import scala.annotation.tailrec

case class GenOptions(
  dim: Int,
  spRows: Int, accRows: Int,
  elemWidth: Int, accWidth: Int,
  chiselArgs: List[String] = Nil
)

object GenOptions {
  def parseArgs(args: Array[String]): GenOptions = {
    @tailrec
    def nextOpt(cur: GenOptions, list: List[String]): GenOptions = {
      list match {
        case Nil => cur
        case "--dim" :: n :: tail =>
          nextOpt(cur.copy(dim = n.toInt), tail)
        case "--sp-rows" :: n :: tail =>
          nextOpt(cur.copy(spRows = n.toInt), tail)
        case "--acc-rows" :: n :: tail =>
          nextOpt(cur.copy(accRows = n.toInt), tail)
        case "--elem-width" :: n :: tail =>
          nextOpt(cur.copy(elemWidth = n.toInt), tail)
        case "--acc-width" :: n :: tail =>
          nextOpt(cur.copy(accWidth = n.toInt), tail)
        case unknown :: tail =>
          nextOpt(cur.copy(chiselArgs = cur.chiselArgs :+ unknown), tail)
      }
    }
    nextOpt(GenOptions(4, 128, 128, 32, 32), args.toList)
  }

}
