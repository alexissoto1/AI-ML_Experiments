<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 10
nchnls = 2
0dbfs = 1.0


instr 1

a1     oscil  0.9, 440,  1
out a1
endin

</CsInstruments>
<CsScore>

f1  0   8192  10   1 .02 .01

</CsScore>
</CsoundSynthesizer>

