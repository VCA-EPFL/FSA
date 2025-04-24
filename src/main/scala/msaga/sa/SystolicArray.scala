package msaga.sa

import chisel3._
import chisel3.util._
import ArithmeticSyntax._

class SystolicArray[E <: Data : Arithmetic, A <: Data : Arithmetic]
(
  rows: Int, cols: Int,
  elemType: E, accType: A,
  macGen: () => MacUnit[E, A],
  cmpGen: () => CmpUnit[A]
) extends Module {

  val io = IO(new Bundle {
    val cmp_ctrl = Flipped(Valid(new CmpControl))
    val pe_ctrl = Flipped(Vec(rows, Valid(new PECtrl)))
    val pe_data = Input(Vec(rows, elemType))
    val acc_out = Output(Vec(cols, Valid(accType)))
  })

  val debug_counter = RegInit(0.U(32.W))
  debug_counter := debug_counter + 1.U
  dontTouch(debug_counter)

  /*
      CMP[0]  -> CMP[1]  -> ... -> CMP[col-1]
       |          |                 |
      PE[0,0] -> PE[0,1] -> ... -> PE[0,col-1]
       |                            |
      ...                          ...
       |                            |
      PE[row-1,0] -> ...        -> PE[row-1, col-1]
  */

  val cmp_array = Seq.fill(cols){ Module(new CMP(accType, cmpGen)) }
  val mesh = Seq.fill(rows) { Seq.fill(cols) { Module(new PE(cols, elemType, accType, macGen)) } }
  val meshT = mesh.transpose

  def pipe_no_reset[T <: Data](in: Valid[T]) = {
    withReset(false.B){ Pipe(in) }
  }

  // left -> right
  cmp_array.foldLeft(io.cmp_ctrl){ (ctrl, cmp) =>
    cmp.io.in_ctrl := ctrl
    pipe_no_reset(cmp.io.out_ctrl)
  }
  for ((row, in_ctrl) <- mesh.zip(io.pe_ctrl)) {
    row.foldLeft(in_ctrl) { (ctrl, pe) => {
      pe.io.in_ctrl := ctrl
      pipe_no_reset(pe.io.out_ctrl)
    }}
  }
  io.pe_data.map(d => {
    val v = Wire(Valid(elemType))
    v.valid := true.B
    v.bits := d
    v
  }).zip(mesh).foreach{case (in_data, row) =>
    row.foldLeft(in_data){ (in, pe) => {
      pe.io.l_input := in
      pipe_no_reset(pe.io.r_output)
    }}
  }

  // up <-> down
  for ((col, cmp) <- meshT.zip(cmp_array)) {
    val cmp_out = pipe_no_reset(cmp.io.d_output)
    // up -> down
    col.foldLeft(cmp_out){ (in, pe) => {
      pe.io.u_input := in
      pipe_no_reset(pe.io.d_output)
    }}
    // down -> up
    val bottom_in = Wire(Valid(accType))
    // TODO: control the bottom input
    bottom_in.valid := true.B
    bottom_in.bits := accType.zero
    col.reverse.foldLeft(bottom_in) { (in, pe) => {
      pe.io.d_input := in
      pipe_no_reset(pe.io.u_output)
    }}
    val cmp_in = pipe_no_reset(col.head.io.u_output)
    cmp.io.d_input := cmp_in
  }

  for ((io_out, pe) <- io.acc_out.zip(meshT.map(_.last))) {
    io_out := pipe_no_reset(pe.io.d_output)
  }

}
