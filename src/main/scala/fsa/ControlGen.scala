package fsa

import chisel3._
import chisel3.util._
import fsa.utils._

import scala.collection.mutable.ListBuffer

class ControlGen(rows: Int) {

  trait FlowDirection

  case object Parallel extends FlowDirection

  case object Upward extends FlowDirection

  case object Downward extends FlowDirection

  // [start, end)
  case class FlowRange(direction: FlowDirection, effStart: Int, repeat: Int) {
    val effEnd = direction match {
      case Parallel => effStart + repeat
      case _ => effStart + repeat + rows - 1
    }

    def update(array: Array[Array[Boolean]], value: Boolean = true): Unit = {
      for (re <- 0 until repeat) {
        for (row <- 0 until rows) {
          direction match {
            case Parallel =>
              array(row)(effStart + re) = value
            case Upward =>
              array(row)(effStart + re + rows - row - 1) = value
            case Downward =>
              array(row)(effStart + re + row) = value
          }
        }
      }
    }
  }

  private val flows: ListBuffer[FlowRange] = ListBuffer()
  private val optimizedFlows: ListBuffer[FlowRange] = ListBuffer()
  private var finalized = false

  def maxCycle: Int = if (flows.isEmpty) 0 else flows.map(_.effEnd).max

  def parallel(start: Int, repeat: Int): Unit = {
    require(!finalized)
    flows += FlowRange(Parallel, start, repeat)
  }

  def flow_up(start: Int, repeat: Int): Unit = {
    require(!finalized)
    flows += FlowRange(Upward, start, repeat)
  }

  def flow_down(start: Int, repeat: Int): Unit = {
    require(!finalized)
    flows += FlowRange(Downward, start, repeat)
  }

  def execution_plan(flow_list: ListBuffer[FlowRange]): Array[Array[Boolean]] = {
    val maxCycle = flow_list.map(_.effEnd).max
    val array = Array.fill(rows, maxCycle)(false)
    flow_list.foreach(_.update(array))
    array
  }

  def raw_execution_plan = execution_plan(flows)

  def opt_execution_plan = execution_plan(optimizedFlows)

  def optimize(): ListBuffer[FlowRange] = {
    val array = execution_plan(flows)
    val tr = array.transpose
    val maxCycle = tr.length
    val ret = ListBuffer[FlowRange]()

    def check_par(cur_col: Int): Int =
      array.view.transpose.slice(cur_col, maxCycle)
        .takeWhile(_.forall(identity)).size

    def check_upward(cur_col: Int): Int = {
      val maxSpan = maxCycle - cur_col
      (0 until maxSpan).find { span =>
        (0 until rows).exists { i =>
          val col = cur_col + span
          val row = rows - i - 1
          val idx = col + i
          idx >= maxCycle || !array(row)(idx)
        }
      }.getOrElse(maxSpan)
    }

    def check_downward(cur_col: Int): Int = {
      val maxSpan = maxCycle - cur_col
      (0 until maxSpan).find { span =>
        (0 until rows).exists { i =>
          val col = cur_col + span
          val row = i
          val idx = col + i
          idx >= maxCycle || !array(row)(idx)
        }
      }.getOrElse(maxSpan)
    }

    var col = 0
    while (col < maxCycle) {
      val par = check_par(col)
      val up = check_upward(col)
      val down = check_downward(col)
      val (flow, step) = (par, up, down) match {
        case (0, 0, 0) =>
          (None, 1)
        case (p, u, d) if p > u && p > d =>
          (Some(FlowRange(Parallel, col, par)), par)
        case (p, u, d) if u > p && u > d =>
          (Some(FlowRange(Upward, col, up)), up + 1)
        case (p, u, d) if d > p && d > u =>
          (Some(FlowRange(Downward, col, down)), down + 1)
        case _ =>
          throw new RuntimeException(f"Invalid Execution Plan\n${planStr(execution_plan(flows))}")
      }
      col += step
      flow.foreach(f => {
        ret += f
        f.update(array, value = false)
      })
    }
    ret
  }

  /**
   *         x1   x2   x3
   *         |    |    |
   * x0 -> r -> r -> r -> ...
   */
  private def shift[T <: Data](x0: T, init: T, upward: Boolean): Vec[T] = {
    val regs = (1 until rows).map(_ => RegInit(init))
    regs.fold(x0) { (prev, cur) => cur := prev; cur }
    if (upward) VecInit((x0 +: regs).reverse) else VecInit(x0 +: regs)
  }

  def generateCtrl(timer: UInt, valid: Bool): Vec[Bool] = {
    import UIntRangeHelper._
    require(!finalized)
    finalized = true
    if (flows.isEmpty) {
      return VecInit(Seq.fill(rows){ false.B })
    }
    optimizedFlows ++= optimize()
    verify()
    //    println(optimizedFlows)
    VecInit(optimizedFlows.map { f =>
      val x0 = valid && timer.between(f.effStart, f.effStart + f.repeat)
      f.direction match {
        case Parallel => VecInit(Seq.fill(rows) {
          x0
        })
        case Upward => shift(x0, false.B, upward = true)
        case Downward => shift(x0, false.B, upward = false)
      }
    }.transpose.map(xi_list => Cat(xi_list.toSeq).orR).toSeq)
  }

  def planStr(plan: Array[Array[Boolean]]): String = plan.map(row => row.map(b => if (b) 1 else 0))
    .map(row => row.mkString(" "))
    .mkString("\n")

  private def verify(): Unit = {
    require(finalized)
    val rawPlan = execution_plan(flows)
    val optPlan = execution_plan(optimizedFlows)
    val pass = rawPlan.zip(optPlan).forall { case (a, b) => a.length == b.length && a.sameElements(b) }

    if (!pass) {
      println("Raw Plan:")
      println(planStr(rawPlan))
      println("New Plan:")
      println(planStr(optPlan))
      require(false)
    }
  }

}

object ControlGen {
  def apply(rows: Int): ControlGen = new ControlGen(rows)
}