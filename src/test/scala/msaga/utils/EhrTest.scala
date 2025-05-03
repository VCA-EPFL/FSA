package msaga.utils

import chisel3._
import chisel3.simulator.{EphemeralSimulator, PeekPokeAPI}
import org.scalatest.flatspec.AnyFlatSpec

class EhrTest extends AnyFlatSpec with PeekPokeAPI {

  val n_ports = 8
  EphemeralSimulator.simulate(new Ehr[UInt](n_ports, UInt(10.W), Some(0.U))) { c =>

    // write (1, 2, 3, ...)
    for (i <- 0 until n_ports) {
      c.io.write(i).valid.poke(true)
      c.io.write(i).bits.poke(i + 1)
    }
    // read (0, 1, 2, ...)
    for (i <- 0 until n_ports) {
      c.io.read(i).expect(i)
    }
    c.clock.step(1)
    // write (n-1, n-2, ...)
    for (i <- 0 until n_ports) {
      c.io.write(i).valid.poke(true)
      c.io.write(i).bits.poke(n_ports - i - 1)
    }
    // read (n, n-1, n-2, ...)
    for (i <- 0 until n_ports) {
      c.io.read(i).expect(n_ports - i)
    }

  }

}
