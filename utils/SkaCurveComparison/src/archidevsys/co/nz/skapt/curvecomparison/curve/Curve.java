package archidevsys.co.nz.skapt.curvecomparison.curve;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import archidevsys.co.nz.skapt.curvecomparison.misc.CurveCompConstants;
import archidevsys.co.nz.skapt.curvecomparison.misc.Utility;

public class Curve
    implements
        CurveCompConstants
{
    private final String name;

    private final File file;

    private ArrayList<DataRow> dataRows;

    private HashMap<Integer, Integer> channelRowCounts;

    public Curve(String pName, File pFileOut)
    {
        name = pName;
        file = pFileOut;

        dataRows = new ArrayList<>(DATA_ROW_COUNT_ESTIMATE);
        channelRowCounts = new HashMap<>(FREQUENCY_CHANNEL_COUNT_ESTIMATE);

        try (FileReader fr = new FileReader(file);
             BufferedReader br = new BufferedReader(fr, BUFFER_SIZE);)
        {
            grabData(br);
        }
        catch (Exception e)
        {
            Utility.abort(e,
                "Problems while attempting to populate the curve [%s]", name);
        }

        System.gc();
    }

    public String getName()
    {
        return name;
    }

    public int getRowCount()
    {
        int RowCount = dataRows.size();

        return RowCount;
    }

    public int getFreqChannelCount(int pFreqChannel)
    {
        int freqChannelCount = channelRowCounts.get(pFreqChannel);

        return freqChannelCount;
    }

    public DataRow getRow(int pIndex)
    {
        DataRow dataRow = dataRows.get(pIndex);

        return dataRow;
    }

    /*
     * ********************* Private Methods ************************
     */
    private void grabData(BufferedReader pBr)
        throws IOException
    {
        String line;
        int rowCount = 0;

        while ((line = pBr.readLine()) != null)
        {
            String[] parts = line.split(REGEX_SPACES);

            int partCount = parts.length;

            if (partCount != ROW_PART_COUNT)
            {
                Utility.abort(
                    "Row %d [%s] of curve [%s] has %d fields not the %d expected!",
                    rowCount, line, name, partCount, ROW_PART_COUNT);
            }

            try
            {
                DataRow dataRow = new DataRow(parts);
                dataRows.add(dataRow);

                incrementChannelRowCount(dataRow);

            }
            catch (NumberFormatException e)
            {
                Utility.abort(e,
                    "Row %d [%s] of curve [%s] has %d a badly formattted number",
                    rowCount, line, name);
            }

            rowCount++;
        }

        dataRows.trimToSize();
    }

    private void incrementChannelRowCount(DataRow dataRow)
    {
        Integer freqChannel = dataRow.getFreqChannel();

        Integer freqChannelCount = 1 +
                        channelRowCounts.getOrDefault(freqChannel, 0);

        channelRowCounts.put(freqChannel, freqChannelCount);
    }
}
