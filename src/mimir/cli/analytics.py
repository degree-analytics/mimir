"""
CLI Analytics Module - Commands for viewing telemetry data and statistics

Provides CLI commands for:
- Basic telemetry statistics (doc_finder stats)
- Detailed analytics and insights (doc_finder analytics)
- Data export functionality (JSON format)
- Time-based filtering and trend analysis
"""

import click
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import sqlite3
from ..core.telemetry import TelemetryManager


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes_val: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


class AnalyticsEngine:
    """Analytics engine for telemetry data analysis."""

    def __init__(self, config: Dict[str, Any] = None, cache_dir: Path = None):
        """Initialize analytics engine."""
        self.config = config or {}
        self.cache_dir = cache_dir or Path(".cache/mimir")
        self.telemetry = TelemetryManager(self.config, self.cache_dir)

    def get_basic_stats(self, time_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get basic telemetry statistics."""
        try:
            with sqlite3.connect(str(self.telemetry.db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Build time filter condition
                time_condition = ""
                params = []
                if time_filter:
                    cutoff_time = self._get_time_cutoff(time_filter)
                    if cutoff_time:
                        time_condition = "WHERE timestamp >= ?"
                        params.append(cutoff_time.timestamp())

                # Total queries
                total_queries = conn.execute(
                    f"""
                    SELECT COUNT(*) as count FROM query_logs {time_condition}
                """,
                    params,
                ).fetchone()["count"]

                if total_queries == 0:
                    return {
                        "total_queries": 0,
                        "message": "No telemetry data found",
                        "time_filter": time_filter,
                    }

                # Average search time
                if time_condition:
                    avg_search_time = (
                        conn.execute(
                            f"""
                        SELECT AVG(search_time_ms) as avg_time FROM query_logs
                        {time_condition} AND search_time_ms IS NOT NULL
                    """,
                            params,
                        ).fetchone()["avg_time"]
                        or 0
                    )
                else:
                    avg_search_time = (
                        conn.execute(
                            """
                        SELECT AVG(search_time_ms) as avg_time FROM query_logs
                        WHERE search_time_ms IS NOT NULL
                    """
                        ).fetchone()["avg_time"]
                        or 0
                    )

                # Cache hit rate
                if time_condition:
                    cache_stats = (
                        conn.execute(
                            f"""
                        SELECT AVG(cache_hit_rate) as avg_hit_rate FROM query_logs
                        {time_condition} AND cache_hit_rate IS NOT NULL
                    """,
                            params,
                        ).fetchone()["avg_hit_rate"]
                        or 0
                    )
                else:
                    cache_stats = (
                        conn.execute(
                            """
                        SELECT AVG(cache_hit_rate) as avg_hit_rate FROM query_logs
                        WHERE cache_hit_rate IS NOT NULL
                    """
                        ).fetchone()["avg_hit_rate"]
                        or 0
                    )

                # Search modes usage
                mode_stats = conn.execute(
                    f"""
                    SELECT search_mode, COUNT(*) as count
                    FROM query_logs {time_condition}
                    GROUP BY search_mode ORDER BY count DESC
                """,
                    params,
                ).fetchall()

                # LLM usage
                if time_condition:
                    llm_stats = conn.execute(
                        f"""
                        SELECT
                            COUNT(*) as total_llm_queries,
                            SUM(llm_tokens_used) as total_tokens,
                            SUM(total_cost_usd) as total_cost,
                            AVG(llm_time_ms) as avg_llm_time
                        FROM query_logs
                        {time_condition} AND llm_tokens_used > 0
                    """,
                        params,
                    ).fetchone()
                else:
                    llm_stats = conn.execute(
                        """
                        SELECT
                            COUNT(*) as total_llm_queries,
                            SUM(llm_tokens_used) as total_tokens,
                            SUM(total_cost_usd) as total_cost,
                            AVG(llm_time_ms) as avg_llm_time
                        FROM query_logs
                        WHERE llm_tokens_used > 0
                    """
                    ).fetchone()

                return {
                    "total_queries": total_queries,
                    "avg_search_time_ms": round(avg_search_time, 2),
                    "cache_hit_rate": round(cache_stats * 100, 1),
                    "search_modes": [
                        {"mode": row["search_mode"], "count": row["count"]}
                        for row in mode_stats
                    ],
                    "llm_usage": {
                        "total_queries": llm_stats["total_llm_queries"] or 0,
                        "total_tokens": llm_stats["total_tokens"] or 0,
                        "total_cost_usd": round(llm_stats["total_cost"] or 0, 6),
                        "avg_time_ms": round(llm_stats["avg_llm_time"] or 0, 2),
                    },
                    "time_filter": time_filter,
                }

        except Exception as e:
            return {"error": f"Failed to get statistics: {e}"}

    def get_detailed_analytics(
        self, time_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed analytics with trends and insights."""
        try:
            with sqlite3.connect(str(self.telemetry.db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Build time filter
                time_condition = ""
                params = []
                if time_filter:
                    cutoff_time = self._get_time_cutoff(time_filter)
                    if cutoff_time:
                        time_condition = "WHERE timestamp >= ?"
                        params.append(cutoff_time.timestamp())

                # Performance trends (hourly data)
                trends = conn.execute(
                    f"""
                    SELECT
                        strftime('%Y-%m-%d %H:00:00', datetime(timestamp, 'unixepoch')) as hour,
                        COUNT(*) as query_count,
                        AVG(search_time_ms) as avg_search_time,
                        AVG(cache_hit_rate) as avg_cache_rate
                    FROM query_logs {time_condition}
                    GROUP BY hour ORDER BY hour DESC LIMIT 24
                """,
                    params,
                ).fetchall()

                # Top queries
                if time_condition:
                    top_queries = conn.execute(
                        f"""
                        SELECT
                            query_text,
                            COUNT(*) as frequency,
                            AVG(result_count) as avg_results,
                            AVG(search_time_ms) as avg_time
                        FROM query_logs
                        {time_condition} AND query_text IS NOT NULL
                        GROUP BY query_text
                        ORDER BY frequency DESC LIMIT 10
                    """,
                        params,
                    ).fetchall()
                else:
                    top_queries = conn.execute(
                        """
                        SELECT
                            query_text,
                            COUNT(*) as frequency,
                            AVG(result_count) as avg_results,
                            AVG(search_time_ms) as avg_time
                        FROM query_logs
                        WHERE query_text IS NOT NULL
                        GROUP BY query_text
                        ORDER BY frequency DESC LIMIT 10
                    """
                    ).fetchall()

                # Model usage analysis
                if time_condition:
                    model_usage = conn.execute(
                        f"""
                        SELECT
                            llm_model,
                            COUNT(*) as usage_count,
                            SUM(llm_tokens_used) as total_tokens,
                            SUM(total_cost_usd) as total_cost,
                            AVG(llm_time_ms) as avg_response_time
                        FROM query_logs
                        {time_condition} AND llm_model IS NOT NULL
                        GROUP BY llm_model ORDER BY usage_count DESC
                    """,
                        params,
                    ).fetchall()
                else:
                    model_usage = conn.execute(
                        """
                        SELECT
                            llm_model,
                            COUNT(*) as usage_count,
                            SUM(llm_tokens_used) as total_tokens,
                            SUM(total_cost_usd) as total_cost,
                            AVG(llm_time_ms) as avg_response_time
                        FROM query_logs
                        WHERE llm_model IS NOT NULL
                        GROUP BY llm_model ORDER BY usage_count DESC
                    """
                    ).fetchall()

                # System performance insights
                performance_insights = self._analyze_performance_patterns(
                    conn, time_condition, params
                )

                return {
                    "performance_trends": [dict(row) for row in trends],
                    "top_queries": [dict(row) for row in top_queries],
                    "model_usage": [dict(row) for row in model_usage],
                    "performance_insights": performance_insights,
                    "time_filter": time_filter,
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            return {"error": f"Failed to get analytics: {e}"}

    def export_data(
        self, format: str = "json", time_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export telemetry data in specified format."""
        try:
            with sqlite3.connect(str(self.telemetry.db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Build time filter
                time_condition = ""
                params = []
                if time_filter:
                    cutoff_time = self._get_time_cutoff(time_filter)
                    if cutoff_time:
                        time_condition = "WHERE timestamp >= ?"
                        params.append(cutoff_time.timestamp())

                # Export query logs
                query_logs = conn.execute(
                    f"""
                    SELECT * FROM query_logs {time_condition} ORDER BY timestamp DESC
                """,
                    params,
                ).fetchall()

                # Convert rows to dict and handle bytes/binary data
                query_logs_data = []
                for row in query_logs:
                    row_dict = dict(row)
                    # Convert any bytes/binary data to string
                    for key, value in row_dict.items():
                        if isinstance(value, bytes):
                            row_dict[key] = value.decode("utf-8", errors="ignore")
                    query_logs_data.append(row_dict)

                export_data = {
                    "export_metadata": {
                        "format": format,
                        "exported_at": datetime.now().isoformat(),
                        "time_filter": time_filter,
                        "total_records": len(query_logs_data),
                    },
                    "query_logs": query_logs_data,
                }

                if format == "json":
                    return export_data
                else:
                    return {"error": f"Format '{format}' not supported"}

        except Exception as e:
            return {"error": f"Failed to export data: {e}"}

    def _get_time_cutoff(self, time_filter: str) -> Optional[datetime]:
        """Get cutoff time for filtering."""
        now = datetime.now()

        if time_filter == "hour":
            return now - timedelta(hours=1)
        elif time_filter == "day":
            return now - timedelta(days=1)
        elif time_filter == "week":
            return now - timedelta(weeks=1)
        elif time_filter == "month":
            return now - timedelta(days=30)
        else:
            return None

    def _analyze_performance_patterns(
        self, conn, time_condition: str, params: List
    ) -> Dict[str, Any]:
        """Analyze performance patterns and generate insights."""
        try:
            # Query performance distribution
            if time_condition:
                perf_stats = conn.execute(
                    f"""
                    SELECT
                        MIN(search_time_ms) as min_time,
                        MAX(search_time_ms) as max_time,
                        AVG(search_time_ms) as avg_time,
                        COUNT(CASE WHEN search_time_ms > 1000 THEN 1 END) as slow_queries
                    FROM query_logs {time_condition} AND search_time_ms IS NOT NULL
                """,
                    params,
                ).fetchone()
            else:
                perf_stats = conn.execute(
                    """
                    SELECT
                        MIN(search_time_ms) as min_time,
                        MAX(search_time_ms) as max_time,
                        AVG(search_time_ms) as avg_time,
                        COUNT(CASE WHEN search_time_ms > 1000 THEN 1 END) as slow_queries
                    FROM query_logs WHERE search_time_ms IS NOT NULL
                """
                ).fetchone()

            # Cache effectiveness
            if time_condition:
                cache_stats = conn.execute(
                    f"""
                    SELECT
                        AVG(cache_hit_rate) as overall_hit_rate,
                        COUNT(CASE WHEN cache_hit_rate > 0.8 THEN 1 END) as high_cache_queries,
                        COUNT(CASE WHEN cache_hit_rate < 0.2 THEN 1 END) as low_cache_queries
                    FROM query_logs {time_condition} AND cache_hit_rate IS NOT NULL
                """,
                    params,
                ).fetchone()
            else:
                cache_stats = conn.execute(
                    """
                    SELECT
                        AVG(cache_hit_rate) as overall_hit_rate,
                        COUNT(CASE WHEN cache_hit_rate > 0.8 THEN 1 END) as high_cache_queries,
                        COUNT(CASE WHEN cache_hit_rate < 0.2 THEN 1 END) as low_cache_queries
                    FROM query_logs WHERE cache_hit_rate IS NOT NULL
                """
                ).fetchone()

            # Generate insights
            insights = []

            if perf_stats["avg_time"] and perf_stats["avg_time"] > 500:
                insights.append(
                    {
                        "type": "performance",
                        "severity": "warning",
                        "message": f"Average search time is {perf_stats['avg_time']:.1f}ms, consider optimization",
                    }
                )

            if perf_stats["slow_queries"] and perf_stats["slow_queries"] > 0:
                insights.append(
                    {
                        "type": "performance",
                        "severity": "info",
                        "message": f"{perf_stats['slow_queries']} queries took over 1 second",
                    }
                )

            if (
                cache_stats["overall_hit_rate"]
                and cache_stats["overall_hit_rate"] < 0.3
            ):
                insights.append(
                    {
                        "type": "cache",
                        "severity": "warning",
                        "message": f"Low cache hit rate ({cache_stats['overall_hit_rate']*100:.1f}%), consider cache optimization",
                    }
                )

            return {
                "performance_distribution": dict(perf_stats) if perf_stats else {},
                "cache_effectiveness": dict(cache_stats) if cache_stats else {},
                "insights": insights,
            }

        except Exception as e:
            return {"error": f"Failed to analyze patterns: {e}"}

    def shutdown(self):
        """Shutdown analytics engine."""
        if self.telemetry:
            self.telemetry.shutdown()


@click.command("stats")
@click.option(
    "--time-filter",
    "-t",
    type=click.Choice(["hour", "day", "week", "month"]),
    help="Filter by time period",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def stats_command(time_filter, format):
    """📊 Show basic telemetry statistics."""
    try:
        # Get cache directory
        cache_dir = Path(".cache/mimir")

        # Initialize analytics engine
        analytics = AnalyticsEngine(cache_dir=cache_dir)

        # Get statistics
        stats = analytics.get_basic_stats(time_filter)

        if "error" in stats:
            click.echo(f"❌ {stats['error']}")
            return

        if format == "json":
            click.echo(json.dumps(stats, indent=2))
            return

        # Text format output
        time_desc = f" ({time_filter})" if time_filter else " (all time)"
        click.echo(f"📊 Mímir Statistics{time_desc}")
        click.echo("=" * 40)

        if stats["total_queries"] == 0:
            click.echo("❌ No telemetry data found")
            click.echo("💡 Run some searches to see statistics")
            return

        # Basic stats
        click.echo(f"Total Queries: {stats['total_queries']}")
        click.echo(
            f"Average Search Time: {format_duration(stats['avg_search_time_ms']/1000)}"
        )
        click.echo(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")

        # Search modes
        if stats["search_modes"]:
            click.echo("\n🔍 Search Modes:")
            for mode in stats["search_modes"][:5]:  # Top 5
                percentage = (mode["count"] / stats["total_queries"]) * 100
                click.echo(f"  {mode['mode']}: {mode['count']} ({percentage:.1f}%)")

        # LLM usage
        llm = stats["llm_usage"]
        if llm["total_queries"] > 0:
            click.echo("\n🤖 LLM Usage:")
            click.echo(f"  AI Queries: {llm['total_queries']}")
            click.echo(f"  Total Tokens: {llm['total_tokens']:,}")
            click.echo(f"  Total Cost: ${llm['total_cost_usd']:.6f}")
            click.echo(
                f"  Avg Response Time: {format_duration(llm['avg_time_ms']/1000)}"
            )

        analytics.shutdown()

    except Exception as e:
        click.echo(f"❌ Error getting statistics: {e}")


@click.command("analytics")
@click.option(
    "--time-filter",
    "-t",
    type=click.Choice(["hour", "day", "week", "month"]),
    help="Filter by time period",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--export", is_flag=True, help="Export detailed data as JSON")
def analytics_command(time_filter, format, export):
    """📈 Show detailed analytics and insights."""
    try:
        # Get cache directory
        cache_dir = Path(".cache/mimir")

        # Initialize analytics engine
        analytics = AnalyticsEngine(cache_dir=cache_dir)

        if export:
            # Export data
            export_data = analytics.export_data("json", time_filter)
            if "error" in export_data:
                click.echo(f"❌ {export_data['error']}")
            else:
                click.echo(json.dumps(export_data, indent=2))
            analytics.shutdown()
            return

        # Get detailed analytics
        data = analytics.get_detailed_analytics(time_filter)

        if "error" in data:
            click.echo(f"❌ {data['error']}")
            analytics.shutdown()
            return

        if format == "json":
            click.echo(json.dumps(data, indent=2))
            analytics.shutdown()
            return

        # Text format output
        time_desc = f" ({time_filter})" if time_filter else " (all time)"
        click.echo(f"📈 Mímir Analytics{time_desc}")
        click.echo("=" * 50)

        # Performance trends
        if data["performance_trends"]:
            click.echo("\n📊 Performance Trends (Last 24 Hours):")
            for trend in data["performance_trends"][:5]:  # Show last 5 hours
                hour = trend["hour"]
                queries = trend["query_count"]
                avg_time = trend["avg_search_time"] or 0
                cache_rate = (trend["avg_cache_rate"] or 0) * 100
                click.echo(
                    f"  {hour}: {queries} queries, {avg_time:.0f}ms avg, {cache_rate:.0f}% cache"
                )

        # Top queries
        if data["top_queries"]:
            click.echo("\n🔍 Most Frequent Queries:")
            for query in data["top_queries"][:5]:
                freq = query["frequency"]
                avg_results = query["avg_results"] or 0
                avg_time = query["avg_time"] or 0
                text = (
                    query["query_text"][:50] + "..."
                    if len(query["query_text"]) > 50
                    else query["query_text"]
                )
                click.echo(
                    f'  "{text}" - {freq}x, {avg_results:.0f} results, {avg_time:.0f}ms'
                )

        # Model usage
        if data["model_usage"]:
            click.echo("\n🤖 LLM Model Usage:")
            for model in data["model_usage"]:
                name = model["llm_model"]
                count = model["usage_count"]
                tokens = model["total_tokens"] or 0
                cost = model["total_cost"] or 0
                time_ms = model["avg_response_time"] or 0
                click.echo(
                    f"  {name}: {count} queries, {tokens:,} tokens, ${cost:.6f}, {time_ms:.0f}ms avg"
                )

        # Performance insights
        insights = data["performance_insights"]
        if insights.get("insights"):
            click.echo("\n💡 Performance Insights:")
            for insight in insights["insights"]:
                emoji = "⚠️" if insight["severity"] == "warning" else "ℹ️"
                click.echo(f"  {emoji} {insight['message']}")

        analytics.shutdown()

    except Exception as e:
        click.echo(f"❌ Error getting analytics: {e}")


# Register commands with the main CLI
def register_analytics_commands(cli_group):
    """Register analytics commands with the main CLI group."""
    cli_group.add_command(stats_command)
    cli_group.add_command(analytics_command)
