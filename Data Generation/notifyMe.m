function notifyMe(msg,subscription_optional)
% Notify via ntfy app on phone, change subscription when needed.
if nargin < 2
    subscription = "boutin-samani-remote-operations-25";
end
msg = sprintf("%s on %s at %s",msg,char(java.net.InetAddress.getLocalHost.getHostName),datestr(now));

system(sprintf('curl -s -d ''%s'' https://ntfy.sh/%s > /dev/null',msg,subscription));
end