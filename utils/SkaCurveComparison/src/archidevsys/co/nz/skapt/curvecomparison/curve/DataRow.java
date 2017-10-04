package archidevsys.co.nz.skapt.curvecomparison.curve;

import archidevsys.co.nz.skapt.curvecomparison.misc.CurveCompConstants;

public class DataRow
    implements
        CurveCompConstants
{
    private final int timeCount;

    private final int freqChannel;

    private final int binNumber;

    private final double valueI;

    private final double valueQ;

    private final double valueU;

    private final double valueV;

    public DataRow(String[] pParts)
        throws NumberFormatException
    {
        timeCount = Integer.parseInt(pParts[0]);
        freqChannel = Integer.parseInt(pParts[1]);
        binNumber = Integer.parseInt(pParts[2]);
        valueI = Double.parseDouble(pParts[3]);
        valueQ = Double.parseDouble(pParts[4]);
        valueU = Double.parseDouble(pParts[5]);
        valueV = Double.parseDouble(pParts[6]);
    }

    public int getTimeCount()
    {
        return timeCount;
    }

    public int getFreqChannel()
    {
        return freqChannel;
    }

    public int getBinNumber()
    {
        return binNumber;
    }

    public double getValueI()
    {
        return valueI;
    }

    public double getValueQ()
    {
        return valueQ;
    }

    public double getValueU()
    {
        return valueU;
    }

    public double getValueV()
    {
        return valueV;
    }

    public double[] getStokesVector()
    {
        double[] stokesArray =
        {
            valueI, valueQ, valueU, valueV
        };

        assert stokesArray.length == STOKES_VECTOR_LENGTH;

        return stokesArray;
    }
}
