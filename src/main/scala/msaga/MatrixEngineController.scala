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
  val valueExecPlan = new AttentionValueExecPlan(DIM)

  // TODO: this is incorrect, need to consider accumulation
  val score_max_cycle = scoreExecPlan.pe_signals.map(_.maxCycle).max
  val cnt1_max_cycle = Seq(score_max_cycle, DIM + 1).max
  val cnt1 = RegInit(0.U(cnt1_max_cycle.U.getWidth.W))
  val value_max_cycle = valueExecPlan.pe_signals.map(_.maxCycle).max
  val cnt2 = RegInit(0.U(value_max_cycle.U.getWidth.W))

  val valid = RegInit(false.B)
  // store information for Q, K, V
  val rs1 = RegEnable(io.in.bits.rs1.asTypeOf(new MatrixRs(SPAD_ROW_ADDR_WIDTH)), io.in.fire)
  // store information for O
  val rs2 = RegEnable(io.in.bits.rs2.asTypeOf(new MatrixRs(ACC_ROW_ADDR_WIDTH)), io.in.fire)

  val loadStationary = RegInit(false.B)
  val kInflight = RegInit(false.B)
  val vInflight = RegInit(false.B)

  /******** ScratchPad Control ********/
  val rd_const = Wire(Bool())
  val rd_const_addr = Wire(chiselTypeOf(rs1.addr))
  val const_rd_rev_in = Wire(Bool())
  val const_rd_rev_out = Wire(Bool())
  val const_rd_delay_out = Wire(Bool())

  io.sp_read.valid := ((loadStationary || kInflight || vInflight) && cnt1 < DIM.U) || rd_const
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
  val t_prop_zero_exp_sum = cnt1.at(2 * DIM + 4)
  // prop zero for P @ V
  val t_prop_zero_p_v = cnt2.between(0, DIM)
  io.cmp_ctrl.valid := Cat(Seq(
    kInflight && (t_update_row_max || t_prop_row_max || t_prop_diff || t_prop_zero_exp_sum),
    vInflight && t_prop_zero_p_v
  )).orR
  io.cmp_ctrl.bits.cmd := Mux1H(Seq(
    (kInflight && t_update_row_max) -> CmpControlCmd.UPDATE,
    (kInflight && t_prop_row_max) -> CmpControlCmd.PROP_MAX,
    (kInflight && t_prop_diff) -> CmpControlCmd.PROP_MAX_DIFF,
    (kInflight && t_prop_zero_exp_sum) -> CmpControlCmd.PROP_ZERO,
    (vInflight && t_prop_zero_p_v) -> CmpControlCmd.PROP_ZERO
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
  val pe_ctrl_score = scoreExecPlan.generatePECtrl(cnt1, kInflight)

  // Value
  val pe_ctrl_value = valueExecPlan.generatePECtrl(cnt2, vInflight)
  io.pe_ctrl := Mux(loadStationary, pe_ctrl_ls, Mux(kInflight, pe_ctrl_score, pe_ctrl_value))

  val cnt1_incr = cnt1 + 1.U
  val cnt2_incr = cnt2 + 1.U
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
      valid := vInflight
      kInflight := false.B
      cnt1 := 0.U
    }.otherwise({
      cnt1 := cnt1_incr
    })
  }
  // k and v can inflight in parallel
  when(vInflight) {
    when(cnt2_incr === value_max_cycle.U) {
      valid := false.B
      vInflight := false.B
      cnt2 := 0.U
    }.otherwise({
      cnt2 := cnt2_incr
    })
  }

  when(io.in.fire) {
    valid := true.B
    loadStationary := io.in.bits.funct7 === ISA.Func.LOAD_STATIONARY
    kInflight := io.in.bits.funct7 === ISA.Func.ATTENTION_SCORE_COMPUTE
    vInflight := io.in.bits.funct7 === ISA.Func.ATTENTION_VALUE_COMPUTE
  }
  io.in.ready := !valid
}
