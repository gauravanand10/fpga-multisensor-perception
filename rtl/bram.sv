// ============================================================================
// bram.sv - inferable block RAM / ROM wrappers (Vivado: RAMB36E1 / RAMB18E1)
//
//   ram_sdp     : simple dual port, 1 write port + 1 read port, registered read (block RAM)
//   ram_sdp_lut : the same in LUT RAM (distributed) - for small tables, saves BRAM
//   ram_sdp_ch  : the same, built from CH-deep pieces (CH=4096 x <=9 bit = one RAMB36,
//                 CH=1024 x <=36 bit = one RAMB36, CH=1024 x <=18 bit = one RAMB18) so
//                 Vivado cannot choose a wasteful layout for odd depths like 12000
//   rom_sp  : block ROM with $readmemh init, registered read
//   rom_lut : small ROM in LUTs (distributed), registered read
// Read latency 1 (address in cycle t -> data in cycle t+1). ram_sdp_ch and rom_sp
// have OREG = 1 for latency 2: an extra register after the piece / cascade output
// mux, so the next stage never sees BRAM clock-to-out + mux in one cycle.
// ============================================================================
`timescale 1ns/1ps

module ram_sdp #(
    parameter int    W     = 8,
    parameter int    D     = 1024,
    parameter int    AW    = $clog2(D),
    parameter string INIT  = ""
)(
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [W-1:0]  wdata,
    input  logic          re,
    input  logic [AW-1:0] raddr,
    output logic [W-1:0]  rdata
);
    (* ram_style = "block" *) logic [W-1:0] mem [D];

    initial begin
        if (INIT != "") $readmemh(INIT, mem);
    end

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule


module ram_sdp_ch #(
    parameter int    W     = 8,
    parameter int    D     = 12000,
    parameter int    CH    = 4096,              // piece depth (power of 2)
    parameter int    AW    = $clog2(D),
    parameter int    OREG  = 0                  // 1: extra output register (latency 2)
)(
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [W-1:0]  wdata,
    input  logic          re,
    input  logic [AW-1:0] raddr,
    output logic [W-1:0]  rdata
);
    localparam int CW = $clog2(CH);
    localparam int NC = (D + CH - 1) / CH;
    localparam int SW = (NC > 1) ? $clog2(NC) : 1;

    wire  [W-1:0]  q [NC];
    logic [SW-1:0] rsel;

    for (genvar c = 0; c < NC; c++) begin : g_piece
        (* ram_style = "block" *) logic [W-1:0] mem [CH];
        logic [W-1:0] qr;
        always_ff @(posedge clk) begin
            if (we && (int'(waddr) >> CW) == c) mem[waddr[CW-1:0]] <= wdata;
            if (re && (int'(raddr) >> CW) == c) qr <= mem[raddr[CW-1:0]];
        end
        assign q[c] = qr;
    end

    always_ff @(posedge clk)
        if (re) rsel <= SW'(int'(raddr) >> CW);

    if (OREG != 0) begin : g_oreg
        always_ff @(posedge clk) rdata <= q[rsel];
    end else begin : g_noreg
        assign rdata = q[rsel];
    end
endmodule


module ram_sdp_lut #(
    parameter int    W     = 8,
    parameter int    D     = 1024,
    parameter int    AW    = $clog2(D)
)(
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [W-1:0]  wdata,
    input  logic          re,
    input  logic [AW-1:0] raddr,
    output logic [W-1:0]  rdata
);
    (* ram_style = "distributed" *) logic [W-1:0] mem [D];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule


module rom_sp #(
    parameter int    W     = 8,
    parameter int    D     = 1024,
    parameter int    AW    = $clog2(D),
    parameter string INIT  = "",
    parameter int    OREG  = 0                  // 1: extra output register (latency 2)
)(
    input  logic          clk,
    input  logic          re,
    input  logic [AW-1:0] addr,
    output logic [W-1:0]  data
);
    (* rom_style = "block" *) logic [W-1:0] mem [D];
    logic [W-1:0] q;

    initial begin
        for (int i = 0; i < D; i++) mem[i] = '0;
        if (INIT != "") $readmemh(INIT, mem);
    end

    always_ff @(posedge clk)
        if (re) q <= mem[addr];

    if (OREG != 0) begin : g_oreg
        always_ff @(posedge clk) data <= q;
    end else begin : g_noreg
        assign data = q;
    end
endmodule


module rom_lut #(
    parameter int    W     = 8,
    parameter int    D     = 32,
    parameter int    AW    = $clog2(D),
    parameter string INIT  = ""
)(
    input  logic          clk,
    input  logic          re,
    input  logic [AW-1:0] addr,
    output logic [W-1:0]  data
);
    (* rom_style = "distributed" *) logic [W-1:0] mem [D];

    initial begin
        for (int i = 0; i < D; i++) mem[i] = '0;
        if (INIT != "") $readmemh(INIT, mem);
    end

    always_ff @(posedge clk)
        if (re) data <= mem[addr];
endmodule
