package archidevsys.co.nz.skapt.curvecomparison.misc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;

public class ProcessLogManager
    implements
        CurveCompConstants
{
    private final static String startYyyymmdd_hhmmss =
        CurveCompContext.getStartYyyymmdd_hhmmss();

    private final File processLogDir;

    private final String summaryLogComment;

    private final String logPrefix;

    private PrintWriter freqChannelLogPw;

    private PrintWriter summaryLogPw;

    private String processLogPrefix;

    private String summaryLogPrefix;

    private String freqChannelLogLeafFilename;

    public ProcessLogManager(
        File pProcessLogDir,
        String pSummaryLogComment,
        String pProcessLogPrefix)
    {
        processLogDir = pProcessLogDir;
        assert processLogDir != null;

        String rawComment = (pSummaryLogComment == null)
            ? EMPTY_STRING
            : pSummaryLogComment;

        summaryLogComment = DOUBLE_QUOTES + rawComment + DOUBLE_QUOTES;

        logPrefix = (pProcessLogPrefix == null)
            ? DEFAULT_LOG_PREFIX
            : pProcessLogPrefix;

        processLogPrefix = logPrefix + "process-";
        summaryLogPrefix = logPrefix + "summary";

        createFreqChannelLogPw();
        createProcessSummaryPw();

    }

    public void logFreqChannel(
        int pFreqChannel,
        int pFreqChannelCount,
        double pFreqChannelFidelity,
        double[] pFreqChannelMse)
    {
        double mseI = pFreqChannelMse[0];
        double mseQ = pFreqChannelMse[1];
        double mseU = pFreqChannelMse[2];
        double mseV = pFreqChannelMse[3];

        freqChannelLogPw.printf("%s %d %d %f %f %f %f %f\n",
            freqChannelLogLeafFilename, pFreqChannel,
            pFreqChannelCount, pFreqChannelFidelity,
            mseI, mseQ, mseU, mseV);
    }

    public void logProcessSummary(
        int pRowCount,
        double pFidelityTotal,
        double[] pMseTotals)
    {
        double mseI = pMseTotals[0];
        double mseQ = pMseTotals[1];
        double mseU = pMseTotals[2];
        double mseV = pMseTotals[3];

        summaryLogPw.printf("%s %s %d %f %f %f %f %f\n",
            freqChannelLogLeafFilename, summaryLogComment,
            pRowCount, pFidelityTotal,
            mseI, mseQ, mseU, mseV);
    }

    /*
     ********************* Private Methods ************************
     */

    private void createFreqChannelLogPw()
    {
        freqChannelLogLeafFilename =
            processLogPrefix + startYyyymmdd_hhmmss + ".log";

        boolean append = false;
        String abortText = "Problem creating [%s]!";

        freqChannelLogPw =
            createPrintWriter(freqChannelLogLeafFilename, append, abortText);
    }

    private void createProcessSummaryPw()
    {
        String leafFilename = summaryLogPrefix + ".log";
        boolean append = true;
        String abortText = "Problem opening [%s]!";
        summaryLogPw = createPrintWriter(leafFilename, append, abortText);
    }

    private PrintWriter createPrintWriter(
        String pLeafFilename,
        boolean pAppend,
        String pAbortText)
    {
        File file = new File(processLogDir, pLeafFilename);
        FileOutputStream fos = null;

        try
        {
            fos = new FileOutputStream(file, pAppend);
        }
        catch (FileNotFoundException e)
        {
            String fullyQualifiedFilename = file.getAbsolutePath();
            Utility.abort(e, pAbortText, fullyQualifiedFilename);
        }

        boolean autoFlush = true;
        PrintWriter pw = new PrintWriter(fos, autoFlush);

        return pw;
    }
}
