package msaga

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import msaga.sa._
import msaga.utils.UIntRangeHelper._

object ConstGen {
  def width = 1
  def ONE = 0.U(width.W)
  def Lg2E = 1.U(width.W)
}

class SpRead()(implicit p: Parameters) extends MSAGABundle {
  val addr = UInt(SPAD_ROW_ADDR_WIDTH.W)
  val rev_sram_out = Bool()
  val delay_sram_out = Bool()
  val rev_delayer_out = Bool()
  val is_constant = Bool()
}

class MatrixEngineController(implicit p: Parameters) extends MSAGAModule {

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new Instruction))
    val sp_read = Valid(new SpRead)
    val cmp_ctrl = Valid(new CmpControl)
    val pe_ctrl = Vec(DIM, Valid(new PECtrl))
  })

  /*
    The attention process is separated into 3 stages:
    1. load stationary (load Q)
    2. attention score: Q @ K / safe online softmax -> P
    3. attention value: P @ V
  */

  val scoreExecPlan = new AttentionScoreExecPlan(DIM)

  val score_max_cycle = scoreExecPlan.pe_signals.map(_.maxCycle).max
  val cnt1_max_cycle = Seq(score_max_cycle, DIM + 1).max
  val cnt1 = RegInit(0.U(cnt1_max_cycle.U.getWidth.W))
  val cnt2 = RegInit(0.U(32.W))

  val valid = RegInit(false.B)
  val rs1 = RegEnable(io.in.bits.rs1.asTypeOf(new MatrixRs(SPAD_ROW_ADDR_WIDTH)), io.in.fire)
  val rs2 = RegEnable(io.in.bits.rs2.asTypeOf(new MatrixRs(ACC_ROW_ADDR_WIDTH)), io.in.fire)

  val kInflight = RegInit(false.B)
  val loadStationary = RegInit(false.B)

  /******** ScratchPad Control ********/
  val rd_const = Wire(Bool())
  val rd_const_addr = Wire(chiselTypeOf(rs1.addr))
  val const_rd_rev_in = Wire(Bool())
  val const_rd_rev_out = Wire(Bool())
  val const_rd_delay_out = Wire(Bool())

  io.sp_read.valid := ((loadStationary || kInflight) && cnt1 < DIM.U) || rd_const
  io.sp_read.bits.addr := Mux(rd_const, rd_const_addr, rs1.addr)
  io.sp_read.bits.rev_sram_out := Mux(rd_const, const_rd_rev_in, rs1.reverseInput)
  io.sp_read.bits.rev_delayer_out := Mux(rd_const, const_rd_rev_out, rs1.reverseOutput)
  io.sp_read.bits.delay_sram_out := Mux(rd_const, const_rd_delay_out, rs1.delayOutput)
  io.sp_read.bits.is_constant := rd_const

  when(io.sp_read.fire) {
    // next row
    rs1.addr := (rs1.addr.zext + rs1.stride).asUInt
  }

  /******** Constant Control ********/
  val t_minus_row_max = cnt1.at(2 * DIM + 1)
  val t_exp_s1 = cnt1.at(2 * DIM + 2)
  val t_exp_sum = cnt1.at(2 * DIM + 4)

  const_rd_rev_in := false.B
  const_rd_rev_out := false.B
  const_rd_delay_out := true.B
  rd_const := kInflight && (t_minus_row_max || t_exp_s1 || t_exp_sum)
  when(t_exp_s1) {
    rd_const_addr := ConstGen.Lg2E
  }.otherwise({
    rd_const_addr := ConstGen.ONE
  })

  /******** CMP Control ********/
  val t_update_row_max = cnt1.between(DIM + 1, 2 * DIM + 1)
  val t_prop_row_max = cnt1.at(2 * DIM + 1)
  val t_prop_diff = cnt1.at(2 * DIM + 2)
  // prop zero for exp sum (1*x+0)
  val t_prop_zero = cnt1.at(2 * DIM + 4)
  io.cmp_ctrl.valid := kInflight && (t_update_row_max || t_prop_row_max || t_prop_diff || t_prop_zero)
  io.cmp_ctrl.bits.cmd := Mux1H(Seq(
    t_update_row_max -> CmpControlCmd.UPDATE,
    t_prop_row_max -> CmpControlCmd.PROP_MAX,
    t_prop_diff -> CmpControlCmd.PROP_MAX_DIFF,
    t_prop_zero -> CmpControlCmd.PROP_ZERO,
  ))

  /******** PE Control ********/
  // Load Stationary
  val pe_ctrl_ls = Wire(Vec(DIM, Valid(new PECtrl)))
  pe_ctrl_ls.foreach(c => {
    c.valid := cnt1.between(1, DIM + 1)
    c.bits := 0.U.asTypeOf(c.bits)
    c.bits.load_reg_li := c.valid
  })

  // Score
  val pe_ctrl_score = Wire(Vec(DIM, Valid(new PECtrl)))
  scoreExecPlan.pe_signals
    .zip(pe_ctrl_score.map(_.bits.getCtrlElements).transpose)
    .foreach { case (gen, ctrlBits) =>
      gen.generateCtrl(cnt1, kInflight).zip(ctrlBits).foreach {
        case (generated, connected) => connected := generated
      }
    }
  pe_ctrl_score.foreach(c => c.valid := c.bits.asUInt.orR)

  io.pe_ctrl := Mux(loadStationary, pe_ctrl_ls, pe_ctrl_score)

  val cnt1_incr = cnt1 + 1.U
  when(loadStationary) {
    when(cnt1_incr === (DIM + 1).U) {
      valid := false.B
      loadStationary := false.B
      cnt1 := 0.U
    }.otherwise({
      cnt1 := cnt1_incr
    })
  }.elsewhen(kInflight) {
    when(cnt1_incr === score_max_cycle.U) {
      valid := false.B
      kInflight := false.B
      cnt1 := 0.U
    }.otherwise({
      cnt1 := cnt1_incr
    })
  }
  when(io.in.fire) {
    valid := true.B
    kInflight := io.in.bits.funct7 === ISA.Func.ATTENTION_SCORE_COMPUTE
    loadStationary := io.in.bits.funct7 === ISA.Func.LOAD_STATIONARY
  }
  io.in.ready := !valid
}
