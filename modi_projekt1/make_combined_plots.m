figure;
plot(out.simout.time, out.simout.signals.values)
hold on
plot(out.simout1.time, out.simout1.signals.values)
xlabel("t(s)")
ylabel("y")
legend("model dyskretny T=5s","model ciągły")
hold off
%print("ciagle_skok0.6_u0=0.6.png", "-dpdf", "-r400")
