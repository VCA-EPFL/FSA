package msaga.svgen

import scala.annotation.tailrec

case class GenOptions(
  dim: Int,
  spRows: Int, accRows: Int,
  mulEW: Int, mulMW: Int,
  addEW: Int, addMW: Int,
  chiselArgs: List[String] = Nil
) {
  def mulWidth = 1 + mulEW + mulMW
  def addWidth = 1 + addEW + addMW
}

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
        case "--mul-ew" :: n :: tail =>
          nextOpt(cur.copy(mulEW = n.toInt), tail)
        case "--mul-mw" :: n :: tail =>
          nextOpt(cur.copy(mulMW = n.toInt), tail)
        case "--add-ew" :: n :: tail =>
          nextOpt(cur.copy(addEW = n.toInt), tail)
        case "--add-mw" :: n :: tail =>
          nextOpt(cur.copy(addMW = n.toInt), tail)
        case unknown :: tail =>
          nextOpt(cur.copy(chiselArgs = cur.chiselArgs :+ unknown), tail)
      }
    }
    nextOpt(GenOptions(4, 128, 128, 5, 10, 5, 10), args.toList)
  }

}
