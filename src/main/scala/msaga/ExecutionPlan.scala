package msaga

import chisel3._
import chisel3.util._
import msaga.sa.PECtrl

trait ExecutionPlan {
  val dim: Int
  val pe_signals = (0 until 9).map(_ => ControlGen(dim)).toList
  val mac :: acc_ui :: load_reg_li :: load_reg_ui :: flow_lr :: flow_ud :: flow_du :: update_reg :: exp2 :: Nil = pe_signals

  def generatePECtrl(timer: UInt, valid: Bool): Vec[Valid[PECtrl]] = {
    val pe_ctrl = Wire(Vec(dim, Valid(new PECtrl)))
    pe_signals
      .zip(pe_ctrl.map(_.bits.getCtrlElements).transpose)
      .foreach { case (gen, ctrlBits) =>
        gen.generateCtrl(timer, valid).zip(ctrlBits).foreach {
          case (generated, connected) => connected := generated
        }
      }
    pe_ctrl.foreach(c => c.valid := c.bits.asUInt.orR)
    pe_ctrl
  }

}

class AttentionScoreExecPlan(val dim: Int) extends ExecutionPlan {
  /****** S = Q @ K ******/
  // stream in K, multiply with Q from bottom left of the SA
  mac.flow_up(1, dim)
  // acc_ui.flow_up(1, DIM) <- we want to acc_di, so ui is false
  flow_lr.flow_up(1, dim)

  /****** Put S back to systolic array ******/
  flow_ud.flow_down(dim + 1, dim)
  load_reg_ui.parallel(2 * dim + 1, 1)

  /****** Flow zero bottom-up for later exp sum ******/
  flow_du.flow_up(dim + 4, dim)

  /****** Staring from the first column, do element-wise ops ******/
  // s = s * 1 + (-m)
  update_reg.flow_down(2 * dim + 2, 1)
  acc_ui.flow_down(2 * dim + 2, 1)
  flow_ud.flow_down(2 * dim + 2, 1)
  flow_lr.flow_down(2 * dim + 2, 1)
  // pass down delta_m; compute (s-m) * log2e in place
  flow_ud.flow_down(2 * dim + 3, 1)
  update_reg.flow_down(2 * dim + 3, 1) // acc_ui is false, not need to control
  flow_lr.flow_down(2 * dim + 3, 1)
  // use pow2 to generate exp
  exp2.flow_down(2 * dim + 4, 1)
  // use mac to compute the sum of exp
  mac.flow_down(2 * dim + 5, 1)
  acc_ui.flow_down(2 * dim + 5, 1)
  flow_lr.flow_down(2 * dim + 5, 1)
}

class AttentionValueExecPlan(val dim: Int) extends ExecutionPlan {
  /****** O = P @ V ******/
  // V enters the SA from upper left
  mac.flow_down(1, dim)
  acc_ui.flow_down(1, dim)
  flow_lr.flow_down(1, dim)

}
