function [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, speed_factor)

    disp("Generating Frame 1")

    [RF_Data, Tstarts] = GenerateRFLinearArray(phantom, transducer, imageopts, 1/speed_factor);
    Frame1 = RFCellToMat(RF_Data, Tstarts, transducer, imageopts);

    phantom.positions = phantom.positions + displacements;

    disp("Generating Frame 2")

    [RF_Data, Tstarts] = GenerateRFLinearArray(phantom, transducer, imageopts, 1/speed_factor);
    Frame2 = RFCellToMat(RF_Data, Tstarts, transducer, imageopts);

end