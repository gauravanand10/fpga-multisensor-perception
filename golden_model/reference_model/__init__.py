"""
reference_model - Python reference model of the complete capstone system

    ESP32-CAM  --SPI-->  Nexys4 DDR FPGA (int8 Tiny U-Net + ToF object distance)  --UART-->  laptop
    VL53L5CX --I2C--> ESP32 --SPI--^

Every stage of the hardware chain has one module here. The FPGA stages are bit-exact: they give
exactly the bytes the SystemVerilog RTL in rtl/ gives (checked in verification/).

    paths.py                    where the model files and the sibling project folders are
    esp32cam_node.py            stage 1  ESP32-CAM: QVGA grey capture -> 160x120 -> CAM1 SPI packet
    tof_node.py                 stage 2  VL53L5CX 8x8 ToF + ESP32 -> TOF1 SPI packet (+ scene simulator)
    spi_link.py                 stage 3  SPI links and the FPGA packet receivers (cam_rx / tof_rx)
    fpga_unet_accelerator.py    stage 4  int8 Tiny U-Net exactly as the FPGA accelerator computes it
    fpga_postprocessing.py      stage 5  objects (connected components) + ToF distance per object
    uart_link.py                stage 6  FPGA -> laptop UART packet (2 Mbaud)
    laptop_host.py              stage 7  laptop: decode the packet, draw "Camera | Depth Map"
    float_reference.py          the original PC system's float U-Net (ONNX Runtime), for comparison
    system_timing.py            latency / throughput budget of the whole chain
    pipeline.py                 EndToEndSystem: all stages chained for one frame
    fpga_model.py               the bit-exact core shared with the RTL testbench generators
"""
