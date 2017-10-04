package archidevsys.co.nz.skapt.curvecomparison.misc;

public class Utility
{
    public static void
        abort(Throwable pThrowable, String pMsgText, Object... pArgs)
    {
        displayAbortMessage(pMsgText, pArgs);
        pThrowable.printStackTrace();
        System.exit(-1);
    }

    public static void abort(String pMsgText, Object... pArgs)
    {
        displayAbortMessage(pMsgText, pArgs);
        System.exit(-1);
    }

    /*
     * ********************* Private Methods ************************
     */

    private static void displayAbortMessage(String pMsgText, Object... pArgs)
    {
        String format = "==> ABORTING " + pMsgText + "\n";
        System.err.printf(format, pArgs);
    }
}
