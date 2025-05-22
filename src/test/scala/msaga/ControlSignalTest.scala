package msaga

import chisel3._
import chisel3.simulator.{EphemeralSimulator, PeekPokeAPI}
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec

class ControlGenWrapper(dim: Int, gen: ControlGen) extends Module {

  val plan = gen.raw_execution_plan
  val maxCycle = plan.head.length

  val io = IO(new Bundle {
    val finish = Output(Bool())
    val success = Output(Bool())
  })

  val timer = RegInit(0.U(log2Up(maxCycle + 1).W))
  val finish = timer === maxCycle.U

  timer := timer + 1.U

  val ctrl = gen.generateCtrl(timer, true.B)

  val plan_ref = VecInit(plan.map(row => VecInit(row.map(_.B))))
  val plan_dut = RegInit(VecInit((0 until dim).map(_ => VecInit(Seq.fill(maxCycle) {
    false.B
  }))))
  val eq = plan_ref.asUInt === plan_dut.asUInt

  for (i <- 0 until dim) {
    plan_dut(i)(timer) := ctrl(i)
  }

  io.finish := RegNext(finish, false.B)
  io.success := RegNext(eq, true.B)

  //  when(finish) {
  //    printf(p"Eq: $eq\n")
  //    printf("Ref:\n")
  //    for (row <- plan_ref) {
  //      for(bit <- row) {
  //        printf(p"$bit ")
  //      }
  //      printf("\n")
  //    }
  //
  //    printf("Dut:\n")
  //    for (row <- plan_dut) {
  //      for(bit <- row) {
  //        printf(p"$bit ")
  //      }
  //      printf("\n")
  //    }
  //  }

}

class ControlSignalTest extends AnyFlatSpec with PeekPokeAPI {

  def test(dim: Int, gen: ControlGen): Unit = {
    EphemeralSimulator.simulate(new ControlGenWrapper(dim, gen)) { c =>
      c.clock.stepUntil(c.io.finish, 1, 6 * dim + 100)
      c.io.success.expect(1)
    }
  }

  it should "handle little overlapped flows" in {
    val dim = 3
    /*
        1 0 1 0 0
        0 1 0 1 0
        0 0 1 0 1
    */
    val gen = ControlGen(dim)
    gen.flow_down(0, 1)
    gen.flow_down(2, 1)
    test(dim, gen)
  }

  it should "handle heavily overlapped flows" in {
    /*
        1 0 1 0 0 0
        0 1 0 1 0 0
        0 0 1 0 1 0
        0 0 0 1 0 1
    */
    val dim = 4
    val gen = ControlGen(dim)
    gen.flow_down(0, 1)
    gen.flow_down(2, 1)
    test(dim, gen)
  }

  it should "merge consecutive flows" in {
    val dim = 3
    val gen = ControlGen(dim)
    /*
        0 0 1 1
        0 1 1 0
        1 1 0 0
    */
    gen.flow_up(0, 1)
    gen.flow_up(1, 1)
    test(dim, gen)
  }

  it should "optimize execution for attention score with small dim" in {
    val dim = 3
    val score = new AttentionScoreExecPlan(dim, dim)
    for (gen <- score.pe_signals.filter(_.maxCycle > 0)) {
      test(dim, gen)
    }
  }

  it should "optimize execution for attention score with medium dim" in {
    val dim = 8
    val score = new AttentionScoreExecPlan(dim, dim)
    for (gen <- score.pe_signals.filter(_.maxCycle > 0)) {
      test(dim, gen)
    }
  }

  it should "optimize execution for attention score with large dim" in {
    val dim = 16
    val score = new AttentionScoreExecPlan(dim, dim)
    for (gen <- score.pe_signals.filter(_.maxCycle > 0)) {
      test(dim, gen)
    }
  }

  it should "optimize execution for attention score with huge dim" in {
    val dim = 64
    val score = new AttentionScoreExecPlan(dim, dim)
    for (gen <- score.pe_signals.filter(_.maxCycle > 0)) {
      test(dim, gen)
    }
  }

}
