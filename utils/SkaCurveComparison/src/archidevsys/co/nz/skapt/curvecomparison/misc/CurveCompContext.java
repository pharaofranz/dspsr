package archidevsys.co.nz.skapt.curvecomparison.misc;

import java.io.File;
import java.util.Calendar;

// XXX Possibly (since jdk1.8) this should be combined into CurveCompConstants
public class CurveCompContext
    implements
        CurveCompConstants
{
    private final static long nanoStart = System.nanoTime();

    private static Calendar calendarStart = Calendar.getInstance();

    private static String startYyyymmdd_hhmmss = String
        .format(YYYYMMDD_HHMMSS_FORMAT, calendarStart);

    public static String getStartYyyymmdd_hhmmss()
    {
        return startYyyymmdd_hhmmss;
    }

    private static File processLogDir = null;

    private static ProcessLogManager processLogManager = null;

    private static String processLogPrefix;

    private static String processLogComment;

    private CurveCompContext()
    {
        // This should never ever be invoked!!!
    }

    public static double getElapsedSeconds()
    {
        long nanoEnd = System.nanoTime();
        double elapsedSeconds = (nanoEnd - nanoStart) / 1e9;

        return elapsedSeconds;
    }

    public static ProcessLogManager getProcessLogManager()
    {
        return processLogManager;
    }

    public static void setProcessLogDir(File pLogDir)
    {
        processLogDir = pLogDir;
    }

    public static void setProcessLogPrefix(String pProcessLogPrefix)
    {
        processLogPrefix = pProcessLogPrefix;
    }

    public static void setProcessLogComment(String pProcessLogComment)
    {
        processLogComment = pProcessLogComment;
    }

    public static void initiateProcessLogging()
    {
        if (processLogDir == null)
        {
            String userDirPathString = System.getProperty("user.dir");
            processLogDir = new File(userDirPathString);
        }

        processLogManager = new ProcessLogManager(processLogDir,
                                                  processLogComment,
                                                  processLogPrefix);
    }
}
