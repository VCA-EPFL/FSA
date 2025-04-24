package msaga

import chisel3._
import chisel3.util._

import msaga.sa.Arithmetic

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

  val out_delay = VecInit(in_data.zipWithIndex.map{ case (d, i) =>
    if (i==0) d else ShiftRegister(d, i)
  })

  val out = Mux(delay, out_delay, in_data)

  io.out := Mux(rev_out, VecInit(out.reverse), out)
}

