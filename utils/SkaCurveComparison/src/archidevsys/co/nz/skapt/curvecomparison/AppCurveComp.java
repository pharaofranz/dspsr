package archidevsys.co.nz.skapt.curvecomparison;

import java.io.File;

import archidevsys.co.nz.skapt.curvecomparison.curve.Curve;
import archidevsys.co.nz.skapt.curvecomparison.curve.CurveComparison;
import archidevsys.co.nz.skapt.curvecomparison.misc.CurveCompConstants;
import archidevsys.co.nz.skapt.curvecomparison.misc.CurveCompContext;
import archidevsys.co.nz.skapt.curvecomparison.misc.ProcessLogManager;
import archidevsys.co.nz.skapt.curvecomparison.misc.Utility;

public class AppCurveComp
    implements
        CurveCompConstants
{
    private double threshold = Double.MAX_VALUE;

    public Curve curveA;

    public Curve curveB;

    public ProcessLogManager processLogManager =
        CurveCompContext.getProcessLogManager();

    public static void main(final String[] pArgs)
    {
        final AppCurveComp appCurveComp = new AppCurveComp();
        appCurveComp.go(pArgs);
    }

    private void go(final String[] pArgs)
    {
        final CommandLine commandLine = new CommandLine(pArgs);
        boolean isValidFlag = commandLine.go();

        Double fidelity = null;

        if (isValidFlag)
        {
            CurveComparison comparison = new CurveComparison(curveA, curveB);
            fidelity = comparison.getFidelity();
            System.out.printf("fidelity: %f\n", fidelity);
        }

        double elapsedSeconds = CurveCompContext.getElapsedSeconds();
        System.out.printf("elapsedSeconds:%6.3f\n", elapsedSeconds);

        int returnCode;

        if (fidelity == null)
        {
            returnCode = -1;
        }
        else
        {
            returnCode = (fidelity < threshold)
                ? 0
                : 1;
        }

        System.exit(returnCode);
    }

    // ------------------- nested classes ------------------------------------

    private class CommandLine
    {
        private int argIndex;

        private int argCount;

        private String[] args;

        private String arg;

        private int argLength;

        private boolean isValidFlag = true;

        private boolean isHelpRequested = false;

        CommandLine(final String[] pArgs)
        {
            args = pArgs;
            argCount = args.length;

            String plural = (argCount == 1)
                ? ""
                : "s";

            System.out.printf("Found %d command line argument%s:\n",
                argCount, plural);

            for (int i = 0; i < argCount; i++)
            {
                arg = args[i];
                System.out.printf("args[%d] [%s]\n", i, arg);
            }

            System.out.println();
        }

        boolean go()
        {
            isHelpRequested = false;

            if (argCount > 0)
            {
                processListOfFlags();

                if (isValidFlag)
                {
                    int remainingArgCount = argCount - argIndex;

                    if (remainingArgCount != 2)
                    {
                        isValidFlag = false;
                    }
                }
            }
            else
            {
                isValidFlag = false;
            }

            if (isHelpRequested || !isValidFlag)
            {
                printHelp();
            }
            else
            {
                CurveCompContext.initiateProcessLogging();
                curveA = makeCurve("curveA");
                curveB = makeCurve("curveB");
            }

            return isValidFlag;
        }

        public Curve makeCurve(String pName)
        {
            String filename = args[argIndex];
            argIndex++;
            File file = getFile(filename);
            Curve curve = new Curve(pName, file);

            return curve;
        }

        private void processListOfFlags()
        {
            argIndex = 0;

            while (argIndex < argCount)
            {
                arg = args[argIndex];
                argLength = arg.length();

                if ((arg.charAt(0) == '-') && (argLength > 1))
                { // processFlagChar can set: isHelpRequested or isValidFlag
                    processFlagChar();

                    if (!isValidFlag)
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }

                argIndex++;
            }
        }

        private void processFlagChar()
        {
            char flagChar = arg.charAt(1);

            switch (flagChar)
            {
                case '-':
                    if (argLength > 2)
                    {
                        processFlagWord();
                    }
                    else
                    {
                        isValidFlag = false;
                    }
                    break;

                case 'd':
                    processLogDirArg();
                    break;

                case 'h':
                    // this will not be actioned until all other flags have been
                    // processed!
                    isHelpRequested = true;
                    isValidFlag = true;
                    break;

                default:
                    isValidFlag = false;
                    System.out.printf("==> Flag [%s] not recognized!\n", arg);
                    break;
            }
        }

        private void processFlagWord()
        {
            String flagWord = arg.substring(2);

            switch (flagWord)
            {
                case "help":
                    // this will not be actioned until all other flags have been
                    // processed!
                    isHelpRequested = true;
                    isValidFlag = true;
                    break;

                case "log-comment":
                    processLogCommentArg();
                    break;

                case "log-prefix":
                    processLogPrefixArg();
                    break;

                case "threshold":
                    processThresholdArg();
                    break;

                default:
                    isValidFlag = false;
                    System.out.printf("==> Flag [%s] not recognized!\n", arg);
                    break;
            }
        }

        private void processLogCommentArg()
        {
            getFlagAgument();

            if (arg != null)
            {
                String processLogComment = arg;
                CurveCompContext.setProcessLogComment(processLogComment);
            }
        }

        @SuppressWarnings("synthetic-access")
        private void processLogDirArg()
        {
            getFlagAgument();

            if (arg != null)
            {

                String processLogDirString = arg;
                File logDir = new File(processLogDirString);

                if (logDir.exists())
                {
                    if (logDir.isDirectory())
                    {
                        if (logDir.canWrite())
                        {
                            CurveCompContext.setProcessLogDir(logDir);
                            isValidFlag = true;
                        }
                        else
                        {
                            Utility.abort(
                                "Directory [%s] cannot be written to!",
                                processLogDirString);
                        }
                    }
                    else
                    {
                        Utility.abort("Directory [%s] is not a directory!",
                            processLogDirString);
                    }
                }
                else
                {
                    Utility.abort("Directory [%s] does not exist!",
                        processLogDirString);
                }
            }
        }

        @SuppressWarnings("synthetic-access")
        private void processLogPrefixArg()
        {
            getFlagAgument();

            if (arg != null)
            {
                String processLogPrefix = arg;
                CurveCompContext.setProcessLogPrefix(processLogPrefix);
            }
        }

        @SuppressWarnings("synthetic-access")
        private void processThresholdArg()
        {
            getFlagAgument();

            if (arg != null)
            {
                threshold = Double.parseDouble(arg);
            }
        }

        private void getFlagAgument()
        {
            argIndex++;

            if (argIndex < argCount)
            {
                isValidFlag = true;
                arg = args[argIndex];
            }
            else
            {
                isValidFlag = false;
                arg = null;
            }
        }

        private File getFile(String pFilename)
        {
            File file = new File(pFilename);

            if (file.exists())
            {
                if (file.isFile())
                {
                    if (file.canRead())
                    {
                        // File appears okay, so far ...
                    }
                    else
                    {
                        Utility.abort("File [%s] cannot be read!", pFilename);
                    }
                }
                else
                {
                    Utility.abort("Path [%s] is not a file!", pFilename);
                }
            }
            else
            {
                Utility.abort("File [%s] does not exist!", pFilename);
            }

            return file;
        }

        private void printHelp()
        {
            System.out.println("Usage:");
            System.out.println("    java <program> <flags> fileA fileB:");
            System.out.println();
            System.out.println("Writes fidelity (Mean Squared Error)");
            System.out.println("       and elapsed in seconds");
            System.out.println("       after listing command line arguments.");
            System.out.println();

            System.out.println("Always sets ReturnCode to zero,");

            System.out
                .println("    unless an error is detected (ReturnCode = -1)");

            System.out.println("    or fidelitythreshold is set");
            System.out.println();
            System.out.println("Command Line Arguments:\n");
            System.out.println("-d             sets process log directory\n");
            System.out.println("-h             prints this help text\n");
            System.out.println("--threshold    sets fidelitythreshold");
            System.out.println(
                "               fidelity < threshold sets ReturnCode to 0");
            System.out.println(
                "               fidelity >= threshold sets ReturnCode to 1\n");
            System.out.println("--help         prints this help text\n");
            System.out.println("--log-comment  adds comment to log file\n");
            System.out.println("--log-prefix   sets process log file prefix");
            System.out.printf("               default is '%s'\n",
                DEFAULT_LOG_PREFIX);
        }
    }
}
