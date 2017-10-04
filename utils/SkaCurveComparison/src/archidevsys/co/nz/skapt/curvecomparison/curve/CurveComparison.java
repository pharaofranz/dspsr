package archidevsys.co.nz.skapt.curvecomparison.curve;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

import archidevsys.co.nz.skapt.curvecomparison.misc.CurveCompContext;
import archidevsys.co.nz.skapt.curvecomparison.misc.ProcessLogManager;
import archidevsys.co.nz.skapt.curvecomparison.misc.Utility;

public class CurveComparison
{
    private static int SVLENGTH = DataRow.STOKES_VECTOR_LENGTH;

    private Curve curveA;

    private Curve curveB;

    private int rowCountA;

    private HashMap<Integer, double[]> channelSeHm;

    private double[] mseTotals = new double[SVLENGTH];

    private ProcessLogManager processLogManager;

    public CurveComparison(Curve pCurveA, Curve pCurveB)
    {
        curveA = pCurveA;
        curveB = pCurveB;

        rowCountA = curveA.getRowCount();
        int rowCountB = curveB.getRowCount();
        int rowCountDelta = rowCountA - rowCountB;

        if (rowCountDelta != 0)
        {
            Utility.abort(
                "File rowcounts differ by %d, rowCountA:%d, rowCountB:%d!",
                rowCountDelta, rowCountA, rowCountB);
        }

        channelSeHm = new HashMap<>(rowCountA);
        processLogManager = CurveCompContext.getProcessLogManager();
    }

    /*
     * Fidelity is the Mean Square Error over all the Stokes' vector components
     * over all rows.
     */
    public double getFidelity()
    {
        calculateChannelSquaredError();
        calculateFreqChannelMse();
        double fidelityTotal = calculateFidelityTotal();

        return fidelityTotal;
    }

    private void calculateChannelSquaredError()
    {
        for (int i = 0; i < rowCountA; i++)
        {
            DataRow rowA = curveA.getRow(i);
            int freqChannelA = rowA.getFreqChannel();
            double[] stokesVectorA = rowA.getStokesVector();

            DataRow rowB = curveB.getRow(i);
            int freqChannelB = rowB.getFreqChannel();
            double[] stokesVectorB = rowB.getStokesVector();

            if (freqChannelA != freqChannelB)
            {
                Utility.abort("Row %d: freqChannelA=%d, but freqChannelB=%d!",
                    rowCountA, freqChannelA, freqChannelB);
            }

            double[] squaredError = new double[SVLENGTH];

            for (int j = 0; j < SVLENGTH; j++)
            {
                double difference = stokesVectorB[j] - stokesVectorA[j];
                squaredError[j] = Math.pow(difference, 2);
            }

            addToChannelSquaredError(freqChannelA, squaredError);
        }
    }

    private void calculateFreqChannelMse()
    {
        Set<Entry<Integer, double[]>> freqChannelSeSet = channelSeHm.entrySet();

        for (Entry<Integer, double[]> freqChannelSe : freqChannelSeSet)
        {
            Integer freqChannel = freqChannelSe.getKey();
            double[] squaredErrors = freqChannelSe.getValue();

            int freqChannelCountA = curveA.getFreqChannelCount(freqChannel);
            int freqChannelCountB = curveB.getFreqChannelCount(freqChannel);

            if (freqChannelCountA != freqChannelCountB)
            {
                Utility.abort(
                    "Row %d: freqChannelCountA=%d, but freqChannelCountB=%d!",
                    rowCountA, freqChannelCountA, freqChannelCountB);
            }

            double[] freqChannelMse = new double[SVLENGTH];
            double freqChannelFidelity = 0;

            for (int i = 0; i < SVLENGTH; i++)
            {
                double mseScalar = squaredErrors[i] / freqChannelCountA;
                freqChannelFidelity += mseScalar;
                freqChannelMse[i] = mseScalar;
                mseTotals[i] += mseScalar;
            }

            processLogManager.logFreqChannel(
                freqChannel,
                freqChannelCountA,
                freqChannelFidelity,
                freqChannelMse);
        }
    }

    private double calculateFidelityTotal()
    {
        double fidelityTotal = 0;

        for (int i = 0; i < SVLENGTH; i++)
        {
            fidelityTotal += mseTotals[i];
        }

        processLogManager.logProcessSummary(
            rowCountA,
            fidelityTotal,
            mseTotals);

        return fidelityTotal;
    }

    /*
     * ********************* Private Methods ************************
     */
    private void addToChannelSquaredError(
        Integer pFreqChannel,
        double[] pSquaredError)
    {
        double[] channelSquaredError = channelSeHm.get(pFreqChannel);

        if (channelSquaredError == null)
        {
            channelSquaredError = pSquaredError;
        }
        else
        {
            for (int i = 0; i < SVLENGTH; i++)
            {
                channelSquaredError[i] += pSquaredError[i];
            }
        }

        channelSeHm.put(pFreqChannel, channelSquaredError);
    }
}
