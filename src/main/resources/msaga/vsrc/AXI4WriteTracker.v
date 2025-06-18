import "DPI-C" function void axi_tracker_dpi(
    input bit aw_fire,
    input int unsigned aw_addr,
    input byte aw_size,
    input byte aw_len,
    input bit w_fire,
    input byte w_data[],
    input bit w_last
);

module AXI4WriteTracker #(
    parameter ADDR_BITS,
    parameter SIZE_BITS,
    parameter LEN_BITS,
    parameter DATA_BITS
)(
    input clock,
    input             aw_fire,
    input [ADDR_BITS-1:0] aw_addr,
    input [SIZE_BITS-1:0] aw_size,
    input [LEN_BITS-1:0]  aw_len,
    input             w_fire,
    input [DATA_BITS-1:0] w_data,
    input             w_last
);

    byte __w_data[(DATA_BITS / 8)-1:0];
    generate
    for (genvar i = 0; i < DATA_BITS / 8; i++) begin
      assign __w_data[i] = w_data[i * 8 +: 8];
    end
    endgenerate

    always @(posedge clock) begin
        axi_tracker_dpi(
            aw_fire,
            aw_addr,
            aw_size,
            aw_len,
            w_fire,
            __w_data,
            w_last
        );
    end

endmodule