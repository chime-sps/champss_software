Example
=======

Here is an example of how to use these scripts to schedule a parallel intensity
and spshuff acquisition during a source transit. Let's pretend it's June 12,
2021, and we want to capture J2111+2016 at 4096 channels.

# Calculate transit time

We first calculate the transit time and row with `source-transit`. This utility
has the position for a few interesting sources encoded within it, so we can
refer to them by name, and then it just makes a call to frb-master's
`calculate-transit` endpoint:

```
(ctl) frb-analysis5: ~$ source-transit j2111 2021-07-13
J2111+2016, 2021-07-13
{
  "peak_transit_time": "2021-07-13 09:44:55.557638 UTC+0000",
  "beam_numbers": [
    57,
    1057,
    2057,
    3057
  ]
}
```

Both source and date are optional. The default source is Crab, just because it's
useful for very early system calibration, while the default date is "today". I
know J2111 will transit in the night, so in UTC it will be "tomorrow",
2021-07-13, and I specify that date as the second argument to `source-transit`.
Make sure you're not calculating last night's transit, and also note the use of
UTC in output!

Next, we can use the `date` utility to translate the time to local site time
(PT), plus determine a start and end window around the transit. Let's say we
want +/- 20 minutes window of data around the transit:

```
(ctl) frb-analysis5: ~$ date --date="2021-07-13 09:44:55.557638 UTC+0000 -20 minutes"
Tue Jul 13 02:24:55 PDT 2021
(ctl) frb-analysis5: ~$ date --date="2021-07-13 09:44:55.557638 UTC+0000 +20 minutes"
Tue Jul 13 03:04:55 PDT 2021
```

The `frb-analysis5` host is set up to keep time in the telescope-local time zone
(PST/PDT), so when we schedule commands to run on it, we'll need to make the
translation. If the SPS cluster were on site and we were running these same
commands from it, we'd keep everything in UTC since the SPS hosts are set up to
use UTC as their time zone.


# Schedule the acquisition

The `at` utility can run a set of commands at a chosen time (similar to Cron,
but in a way that works better for our purpose), so let's use it. (Run `man at`
if you want to learn more about what it can do.)

`at` takes a time specification and then reads from its stdin the sequence of
commands to run. We use a little shell trick to feed it to it (`<<EOF ... EOF`),
like this:

```
at 02:24:55 <<EOF
source  ~/.local/share/virtualenvs/ctl/bin/activate
start-sps 4096
start-intensity 57
EOF
```
Equivalently, you can run `at 02:24:55`, then at the prompt type the three commands, and press CTRL-D.

Note that we first activate the virtual environment in which we can find the
scripts. (Adjust as needed for your case.) For `start-sps`, we give it the
desired number of channels, that is 4096. (If left out, the default nchan is
1024.) For `start-intensity`, we must give the target row, which we get from the
output of `source-transit`, i.e., 57. (If left out, the default row is 59, i.e.,
Crab.) We do not need to give the target row to `start-sps` because it acquires
all eight rows 56-63.

At end time, we do similar steps with `stop-` scripts:
```
at 03:04:55 <<EOF
source  ~/.local/share/virtualenvs/ctl/bin/activate
stop-sps
stop-intensity 57
EOF
```
In this case, only `stop-intensity` needs an argument -- this has to match the row used to start the acquisition!

# Mailed output

The `at` command will mail the output of its run to the user who ran it. On
frb-analysis5 you can read it if you run `mailx` from the terminal. This is one
of the early Unix terminal utilities, so it's not exactly user-friendly, but is
simple enough to use (`man mailx` or look for online tutorials about "mailx",
e.g., [from Solaris][sol]).

Alternatively, you can set up email forwarding by putting your preferred email
address into the file named `.forward` in your home directory.

# Accessing data

Because intensity acquisitions are saved into a dedicated directory on the
frb-archiver, we need to move that data into the SPS area, in format in which
the pipeline `quant` can find them. The below paths are all unique to
frb-analysis5 -- look at the actual mounts if you need to find out the true
paths on frb-archiver.

The directory for acquisitions is
`/data/frb-archiver/chime/intensity/raw/acq_data`, and L1 will stream the
intensity data into a named subdirectory that is given in the "start-intensity"
call. By default, `start-intensity` will name the acquisition
`sps_YYYYmmdd_HHMM`. (You can specify your own name as the second argument to
`start-intensity`.) We can check for it:

```
frb-analysis5: $ ls -ld /data/frb-archiver/chime/intensity/raw/acq_data/sps*
drwxrwxrwx. 6 root root 6 Jul 13 01:29 /data/frb-archiver/chime/intensity/raw/acq_data/sps_20210713-0738/
drwxrwxrwx. 6 root root 6 May  8  2020 /data/frb-archiver/chime/intensity/raw/acq_data/sps_incoherent_dump_1588948316/
drwxrwxrwx. 6 root root 6 May  8  2020 /data/frb-archiver/chime/intensity/raw/acq_data/sps_incoherent_dump_1588948755/
```

I find it easiest to copy the files using `rsync`:
```
rsync -rtlv --partial /data/frb-archiver/chime/intensity/raw/acq_data/sps_20210713-0738/beam_* /data/sps-data/intensity/2021/07/13
```

Finally, don't forget to remove the acquisition directory (you'll need sudo because it's owned by `root`):
```
sudo rm -r /data/frb-archiver/chime/intensity/raw/acq_data/sps_1919_1922_20210713-0738
```
(Be careful with what you're deleting -- no wildcards so you don't end up removing long-term data by accident!)


# Copying to SPS test bed

Similar procedure will have to happen to copy the data from frb-archiver to the
slow pulsar test bed. For instance, to copy the spshuff acquisitions from June
and July 2021 to sps-st1, I do:

```
sps-st1:~$ rsync -rltv --partial frb-analysis5:/data/chime/sps/raw/2021/{06,07} /data/chime/sps/raw/2021
....
```

For intensity, it depends if I'm copying what's already been copied into the SPS
disk area, or if I'm copying straight from `acq_data`, where it was written by L1:
```
# If data is still in `acq_data` where it was written by L1:
# Note I specify the destination date directory -- make sure you're using the right one!
sps-st1:~$ rsync -rltv --partial frb-analysis5:/data/frb-archiver/chime/intensity/raw/acq_data//sps_20210713-0738/beam_* /data/chime/intensity/raw/2021/07/13

# If data has already been moved to the SPS area, such as the
# acquisitions done while the cluster was being moved from UBC to McGill
# Note that I'm narrowing down to just the two months to limit
# the number of files that need to be checked
sps-st1:~$ rsync -rltv --partial frb-analysis5:/data/chime/intensity/raw/2021/{06,07} /data/chime/intensity/raw/2021
```

[sol]: https://docs.oracle.com/cd/E19683-01/806-7612/mail-1/index.html
