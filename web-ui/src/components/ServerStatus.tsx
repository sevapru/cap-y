import { useState, useEffect } from 'react';

interface ServerInfo {
  group: string;
  port: number;
  status: string;
  ok: boolean;
  desc: string;
}

interface StatusResponse {
  profile: string;
  backends: Record<string, ServerInfo>;
}

interface ServerStatusProps {
  gatewayUrl: string;
}

const STATUS_COLOR: Record<string, string> = {
  ready: 'bg-nv-green',
  ok: 'bg-nv-green',
  initializing: 'bg-yellow-500',
  unreachable: 'bg-red-500',
  unknown: 'bg-text-tertiary',
};

export const ServerStatus: React.FC<ServerStatusProps> = ({ gatewayUrl }) => {
  const [data, setData] = useState<StatusResponse | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${gatewayUrl}/status`, { signal: AbortSignal.timeout(3000) });
        if (!r.ok) throw new Error(`Gateway returned ${r.status}`);
        setData(await r.json());
        setLastUpdated(new Date());
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Gateway unreachable');
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [gatewayUrl]);

  const grouped = data
    ? Object.entries(data.backends).reduce<Record<string, [string, ServerInfo][]>>(
        (acc, [name, info]) => {
          if (!acc[info.group]) acc[info.group] = [];
          acc[info.group].push([name, info]);
          return acc;
        },
        {}
      )
    : {};

  return (
    <div className="flex flex-col h-full bg-surface overflow-auto p-4 gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <h2 className="text-sm font-bold font-display text-text-primary tracking-widest uppercase">
          Servers
        </h2>
        <div className="flex items-center gap-2">
          {data && (
            <span className="text-xs text-text-tertiary font-display uppercase tracking-wide">
              {data.profile}
            </span>
          )}
          {lastUpdated && (
            <span className="text-xs text-text-tertiary">
              {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Error state */}
      {error && !data && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-surface-raised border border-surface-border">
          <div className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
          <span className="text-xs text-text-secondary font-mono">{error}</span>
        </div>
      )}

      {/* Grouped server list */}
      {Object.entries(grouped).map(([group, servers]) => (
        <div key={group} className="flex flex-col gap-1.5">
          <span className="text-xs font-display font-medium text-text-tertiary tracking-widest uppercase px-1">
            {group}
          </span>
          <div className="flex flex-col gap-1">
            {servers.map(([name, info]) => {
              const colorClass = STATUS_COLOR[info.status] ?? STATUS_COLOR.unknown;
              const docsUrl = `${gatewayUrl}/${group}/${name}/docs`;
              return (
                <a
                  key={name}
                  href={docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-3 py-2 rounded-md bg-surface-raised border border-surface-border hover:border-accent/30 hover:bg-surface-overlay transition-all group"
                >
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${colorClass} ${info.ok ? 'shadow-sm' : ''}`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-display font-medium text-text-primary">
                        {name}
                      </span>
                      <span className="text-xs text-text-tertiary font-mono">:{info.port}</span>
                    </div>
                    {info.desc && (
                      <span className="text-xs text-text-tertiary truncate block">{info.desc}</span>
                    )}
                  </div>
                  <span className={`text-xs font-display font-medium flex-shrink-0 ${info.ok ? 'text-nv-green' : 'text-text-tertiary'}`}>
                    {info.status}
                  </span>
                </a>
              );
            })}
          </div>
        </div>
      ))}

      {/* Empty state */}
      {data && Object.keys(grouped).length === 0 && (
        <p className="text-xs text-text-tertiary text-center py-4">No backends registered for this profile.</p>
      )}
    </div>
  );
};
