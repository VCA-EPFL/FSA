package fsa

import chisel3._
import chisel3.util._
import fsa.arithmetic.Arithmetic

/*
    By default, X(i) is delayed by i cycle.
    x0 -> x0
    x1 -> r -> x1
    x2 -> r -> r  -> x2
*/
class InputDelayer[E <: Data : Arithmetic](rows: Int, elemType: E) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Valid(new Bundle {
      val data = Vec(rows, elemType)
      val rev_input = Bool()
      val delay_output = Bool()
      val rev_output = Bool()
    }))
    val out = Output(Vec(rows, elemType))
  })

  val rev_out_r = RegEnable(io.in.bits.rev_output, io.in.fire)
  val rev_out = Mux(io.in.valid, io.in.bits.rev_output, rev_out_r)

  val delay_r = RegEnable(io.in.bits.delay_output, io.in.fire)
  val delay = Mux(io.in.valid, io.in.bits.delay_output, delay_r)

  val in_data = Mux(io.in.bits.rev_input, VecInit(io.in.bits.data.reverse), io.in.bits.data)

  val out_delay = VecInit(in_data.zipWithIndex.map { case (d, i) =>
    if (i == 0) d else ShiftRegister(d, i)
  })

  val out = Mux(delay, out_delay, in_data)

  io.out := Mux(rev_out, VecInit(out.reverse), out)
}

class OutputDelayer[A <: Data : Arithmetic](cols: Int, accType: A) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(cols, accType))
    val out = Output(Vec(cols, accType))
  })

  val impl = Module(new InputDelayer(cols, accType))
  impl.io.in.valid := true.B
  impl.io.in.bits.data := io.in
  impl.io.in.bits.rev_input := true.B
  impl.io.in.bits.rev_output := true.B
  impl.io.in.bits.delay_output := true.B

  io.out := impl.io.out
}

